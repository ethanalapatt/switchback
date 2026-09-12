"""Cost-based draft-length controller.

Chooses the next draft length from ``{0, 1, 2, 4, 8}``, where ``0`` is bypass:
ordinary target-only decoding. The decision is made **before** the block's
candidates are sampled, using only the committed prefix, a calibration profile,
and observations from earlier blocks of the same request. That ordering is what
keeps the distribution argument in ``docs/correctness.md`` section 3 intact.

Deliberately simple and inspectable (SPEC.md section 4.4): no learned policy, no
reinforcement learning, and every number that goes into a decision can be
printed. What it is *not* is a guarantee of no slowdown -- cost estimates can be
wrong, workloads shift, and startup amortization is a guess about a length the
controller cannot know.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from switchback.provenance import sha256_bytes
from switchback.types import CONTROLLER_ACTIONS, BlockObservation, ConfigError

PROFILE_SCHEMA_VERSION = 1

# SPEC.md section 4.4. A positive draft length must beat target-only decoding by
# this margin before it is chosen. It is a design setting, not a promised gain.
DEFAULT_MARGIN = 0.05
DEFAULT_EWMA_ALPHA = 0.2

# Context buckets in cached tokens, matching the controlled-prompt lengths in
# SPEC.md section 9.1 so calibration and evaluation use the same axis.
CONTEXT_BUCKETS: tuple[int, ...] = (0, 256, 1024, 2048)


def context_bucket(cached_tokens: int) -> int:
    """Index of the bucket ``cached_tokens`` falls into."""
    bucket = 0
    for index, edge in enumerate(CONTEXT_BUCKETS):
        if cached_tokens >= edge:
            bucket = index
    return bucket


@dataclass(frozen=True)
class AcceptanceTable:
    """Conditional acceptance by candidate position, from calibration counts.

    ``successes[j]`` and ``trials[j]`` count position ``j`` **only among blocks
    that reached it**. A block rejected at position 2 contributes successes at 0
    and 1, a failure at 2, and *nothing at all* at 3 and beyond: those positions
    were never evaluated. Recording them as failures is the classic censoring
    mistake, and it biases long draft lengths downward exactly where the
    controller is deciding whether to use them.
    """

    successes: tuple[int, ...]
    trials: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.successes) != len(self.trials):
            raise ConfigError("acceptance successes and trials must have the same length")
        for index, (wins, tries) in enumerate(zip(self.successes, self.trials, strict=True)):
            if wins < 0 or tries < 0:
                raise ConfigError(f"negative acceptance count at position {index}")
            if wins > tries:
                raise ConfigError(
                    f"position {index}: {wins} successes out of {tries} trials is impossible"
                )

    @property
    def positions(self) -> int:
        return len(self.trials)

    def rate(self, position: int) -> float:
        """Beta(1, 1) smoothed conditional acceptance at ``position``.

        A position with no calibration data returns the prior mean of 0.5 rather
        than 0 or 1, so an unobserved long position neither kills nor endorses a
        long draft length on no evidence.
        """
        if position < 0:
            raise ConfigError(f"position must be >= 0, got {position}")
        if position >= self.positions:
            return 0.5
        return (self.successes[position] + 1) / (self.trials[position] + 2)

    def merged(self, successes: Sequence[int], trials: Sequence[int]) -> AcceptanceTable:
        """Combine calibration counts with request-local observations."""
        width = max(self.positions, len(trials))

        def total(base: tuple[int, ...], extra: Sequence[int], index: int) -> int:
            return (base[index] if index < len(base) else 0) + (
                extra[index] if index < len(extra) else 0
            )

        return AcceptanceTable(
            successes=tuple(total(self.successes, successes, index) for index in range(width)),
            trials=tuple(total(self.trials, trials, index) for index in range(width)),
        )


@dataclass(frozen=True)
class CostTable:
    """Measured stage costs in nanoseconds, from calibration.

    ``target_forward_ns`` is keyed by verification width. On memory-bound
    hardware it is nearly flat, which is the whole reason speculation can pay;
    the controller reads the real shape rather than assuming it.
    """

    target_forward_ns: dict[int, float]
    draft_forward_ns: float
    block_overhead_ns: float
    draft_prefill_ns: float
    median_remaining_tokens: float

    def __post_init__(self) -> None:
        if not self.target_forward_ns:
            raise ConfigError("cost table has no target forward measurements")
        for width, cost in self.target_forward_ns.items():
            if width < 1 or cost <= 0:
                raise ConfigError(f"invalid target forward cost at width {width}: {cost}")
        for name in ("draft_forward_ns", "draft_prefill_ns", "median_remaining_tokens"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        if self.block_overhead_ns < 0:
            raise ConfigError("block_overhead_ns must be non-negative")

    def target_at_width(self, width: int) -> float:
        """Cost of one target forward at ``width``, interpolated if unmeasured."""
        if width in self.target_forward_ns:
            return self.target_forward_ns[width]
        widths = sorted(self.target_forward_ns)
        if width < widths[0]:
            return self.target_forward_ns[widths[0]]
        if width > widths[-1]:
            return self.target_forward_ns[widths[-1]]
        lower = max(value for value in widths if value <= width)
        upper = min(value for value in widths if value >= width)
        if lower == upper:
            return self.target_forward_ns[lower]
        span = upper - lower
        weight = (width - lower) / span
        return self.target_forward_ns[lower] * (1 - weight) + self.target_forward_ns[upper] * weight


@dataclass(frozen=True)
class CostProfile:
    """Everything the controller was calibrated with, frozen and hashable.

    The hash is recorded before evaluation so a reviewer can confirm the
    controller was not retuned after seeing held-out results.
    """

    schema_version: int = PROFILE_SCHEMA_VERSION
    acceptance: AcceptanceTable = field(default_factory=lambda: AcceptanceTable((), ()))
    cost: CostTable = field(default_factory=lambda: CostTable({1: 1.0}, 1.0, 0.0, 1.0, 1.0))
    actions: tuple[int, ...] = CONTROLLER_ACTIONS
    margin: float = DEFAULT_MARGIN
    ewma_alpha: float = DEFAULT_EWMA_ALPHA
    calibration_source: str = ""
    source_commit: str | None = None

    def __post_init__(self) -> None:
        if 0 not in self.actions:
            raise ConfigError("the action set must include 0 (bypass)")
        if sorted(self.actions) != list(self.actions):
            raise ConfigError("actions must be sorted")
        if not 0.0 <= self.margin < 1.0:
            raise ConfigError(f"margin must be in [0, 1), got {self.margin}")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ConfigError(f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}")

    def as_dict(self) -> dict[str, Any]:
        document = asdict(self)
        document["cost"]["target_forward_ns"] = {
            str(width): cost for width, cost in self.cost.target_forward_ns.items()
        }
        document["actions"] = list(self.actions)
        return document

    def sha256(self) -> str:
        """Stable hash of the frozen profile."""
        return sha256_bytes(json.dumps(self.as_dict(), sort_keys=True).encode())


def profile_from_dict(document: dict[str, Any]) -> CostProfile:
    """Rebuild a profile written by :meth:`CostProfile.as_dict`."""
    if document.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ConfigError(
            f"profile schema_version {document.get('schema_version')!r} is not "
            f"{PROFILE_SCHEMA_VERSION}"
        )
    cost = dict(document["cost"])
    cost["target_forward_ns"] = {
        int(width): float(value) for width, value in cost["target_forward_ns"].items()
    }
    return CostProfile(
        schema_version=document["schema_version"],
        acceptance=AcceptanceTable(
            successes=tuple(document["acceptance"]["successes"]),
            trials=tuple(document["acceptance"]["trials"]),
        ),
        cost=CostTable(**cost),
        actions=tuple(document["actions"]),
        margin=float(document["margin"]),
        ewma_alpha=float(document["ewma_alpha"]),
        calibration_source=document.get("calibration_source", ""),
        source_commit=document.get("source_commit"),
    )


@dataclass(frozen=True)
class ControllerState:
    """Everything the controller is allowed to see.

    Invariant I9 is enforced by this schema rather than by discipline: there is
    no field for a dataset name, a prompt id, an expected answer, or the length
    the request will turn out to be. A decision cannot depend on information the
    state cannot carry.
    """

    cached_tokens: int
    remaining_budget: int
    draft_initialized: bool
    blocks_observed: int

    def __post_init__(self) -> None:
        if self.cached_tokens < 0:
            raise ConfigError("cached_tokens must be >= 0")
        if self.remaining_budget < 1:
            raise ConfigError("remaining_budget must be >= 1")
        if self.blocks_observed < 0:
            raise ConfigError("blocks_observed must be >= 0")

    @property
    def context_bucket(self) -> int:
        return context_bucket(self.cached_tokens)


@dataclass(frozen=True)
class ActionEstimate:
    """The controller's arithmetic for one candidate action, kept inspectable."""

    action: int
    expected_tokens: float
    block_cost_ns: float
    cost_per_token_ns: float
    all_accepted_probability: float
    startup_ns: float
    feasible: bool
    reason: str = ""


@dataclass(frozen=True)
class ControllerDecision:
    """A chosen action plus the estimates that produced it."""

    action: int
    reason: str
    estimates: tuple[ActionEstimate, ...]
    target_only_cost_ns: float
    sticky_bypass: bool


class Controller:
    """Minimal controller interface used by the decoder."""

    def choose(self, state: ControllerState) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def observe(self, block: BlockObservation) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class FixedController(Controller):
    """Always proposes the same draft length. The ``fixed_g`` baselines.

    It still counts its decisions, so a fixed engine reports the same counters
    as the adaptive one and the two rows in a report are directly comparable.
    ``FixedController(0)`` is target-only decoding, where every decision is a
    bypass by construction.
    """

    gamma: int
    decisions: int = 0
    bypass_decisions: int = 0

    def __post_init__(self) -> None:
        if self.gamma < 0:
            raise ConfigError(f"fixed gamma must be >= 0, got {self.gamma}")

    def choose(self, state: ControllerState) -> int:
        self.decisions += 1
        action = min(self.gamma, max(0, state.remaining_budget - 1))
        if action == 0:
            self.bypass_decisions += 1
        return action

    def observe(self, block: BlockObservation) -> None:
        return None

    def reset(self) -> None:
        self.decisions = 0
        self.bypass_decisions = 0


@dataclass
class CostController(Controller):
    """Picks the action with the lowest estimated cost per committed token.

    Request-local state -- acceptance observations, cost EWMAs, and the sticky
    bypass flag -- is cleared by :meth:`reset` between requests, so no adaptive
    state leaks from one request into the next (invariant I10).
    """

    profile: CostProfile
    allow_bypass: bool = True
    _successes: list[int] = field(default_factory=list, repr=False)
    _trials: list[int] = field(default_factory=list, repr=False)
    _observed_block_ns: dict[int, float] = field(default_factory=dict, repr=False)
    _bypassed: bool = field(default=False, repr=False)
    _decisions: int = field(default=0, repr=False)
    _bypass_decisions: int = field(default=0, repr=False)
    last_decision: ControllerDecision | None = field(default=None, repr=False)

    # --- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self._successes = []
        self._trials = []
        self._observed_block_ns = {}
        self._bypassed = False
        self._decisions = 0
        self._bypass_decisions = 0
        self.last_decision = None

    @property
    def decisions(self) -> int:
        return self._decisions

    @property
    def bypass_decisions(self) -> int:
        return self._bypass_decisions

    @property
    def bypassed(self) -> bool:
        return self._bypassed

    # --- estimation --------------------------------------------------------

    def acceptance(self) -> AcceptanceTable:
        """Calibration counts combined with this request's observations."""
        return self.profile.acceptance.merged(self._successes, self._trials)

    def expected_tokens(self, action: int, table: AcceptanceTable) -> tuple[float, float]:
        """``E(g) = 1 + sum_j prod_{k<=j} a_k`` and the all-accepted probability."""
        expected = 1.0
        running = 1.0
        for position in range(action):
            running *= table.rate(position)
            expected += running
        return expected, running if action else 1.0

    def block_cost_ns(self, action: int, all_accepted: float) -> float:
        """Modelled cost of one block at ``action``.

        Includes the draft's autoregressive steps, the target verification at
        width ``g + 1``, block overhead, and the catch-up forward weighted by
        the probability that every candidate is accepted. The catch-up is not a
        rounding error: on this hardware it is a full draft forward, paid on
        most blocks when acceptance is high.
        """
        cost = self.profile.cost
        if action == 0:
            return cost.target_at_width(1) + cost.block_overhead_ns
        observed = self._observed_block_ns.get(action)
        if observed is not None:
            return observed
        return (
            action * cost.draft_forward_ns
            + cost.target_at_width(action + 1)
            + cost.block_overhead_ns
            + all_accepted * cost.draft_forward_ns
        )

    def startup_ns(self, action: int, state: ControllerState) -> float:
        """Draft prefill, amortized over the length this request is likely to run.

        Charged only when the draft is not yet resident and the action drafts.
        The horizon is the smaller of the remaining budget and the calibration
        median completion length, because a request that stops early never
        recovers the prefill. This estimate is wrong whenever EOS arrives
        sooner, which is why the natural-stop cohort is evaluated separately.
        """
        if action == 0 or state.draft_initialized:
            return 0.0
        horizon = max(
            1.0, min(float(state.remaining_budget), self.profile.cost.median_remaining_tokens)
        )
        return self.profile.cost.draft_prefill_ns / horizon

    def estimate(self, state: ControllerState) -> ControllerDecision:
        """Full arithmetic for every action, without changing any state."""
        table = self.acceptance()
        cap = state.remaining_budget - 1
        estimates: list[ActionEstimate] = []
        target_only_cost = self.block_cost_ns(0, 1.0)

        for action in self.profile.actions:
            if action == 0:
                estimates.append(
                    ActionEstimate(
                        action=0,
                        expected_tokens=1.0,
                        block_cost_ns=target_only_cost,
                        cost_per_token_ns=target_only_cost,
                        all_accepted_probability=1.0,
                        startup_ns=0.0,
                        feasible=True,
                        reason="target-only step",
                    )
                )
                continue
            if action > cap:
                estimates.append(
                    ActionEstimate(
                        action=action,
                        expected_tokens=0.0,
                        block_cost_ns=math.inf,
                        cost_per_token_ns=math.inf,
                        all_accepted_probability=0.0,
                        startup_ns=0.0,
                        feasible=False,
                        reason=f"clipped: only {state.remaining_budget} tokens of budget remain",
                    )
                )
                continue
            expected, all_accepted = self.expected_tokens(action, table)
            startup = self.startup_ns(action, state)
            cost = self.block_cost_ns(action, all_accepted)
            per_token = cost / expected + startup
            estimates.append(
                ActionEstimate(
                    action=action,
                    expected_tokens=expected,
                    block_cost_ns=cost,
                    cost_per_token_ns=per_token,
                    all_accepted_probability=all_accepted,
                    startup_ns=startup,
                    feasible=True,
                    reason="",
                )
            )

        drafting = [item for item in estimates if item.action > 0 and item.feasible]
        if not drafting:
            return ControllerDecision(
                action=0,
                reason="no draft length fits the remaining budget",
                estimates=tuple(estimates),
                target_only_cost_ns=target_only_cost,
                sticky_bypass=False,
            )
        best = min(drafting, key=lambda item: item.cost_per_token_ns)
        threshold = (1.0 - self.profile.margin) * target_only_cost
        if best.cost_per_token_ns <= threshold:
            return ControllerDecision(
                action=best.action,
                reason=(
                    f"g={best.action} at {best.cost_per_token_ns:.0f} ns/token beats "
                    f"target-only {target_only_cost:.0f} by more than "
                    f"{100 * self.profile.margin:.0f}%"
                ),
                estimates=tuple(estimates),
                target_only_cost_ns=target_only_cost,
                sticky_bypass=False,
            )
        return ControllerDecision(
            action=0,
            reason=(
                f"best draft length g={best.action} costs "
                f"{best.cost_per_token_ns:.0f} ns/token against target-only "
                f"{target_only_cost:.0f}; margin not met"
            ),
            estimates=tuple(estimates),
            target_only_cost_ns=target_only_cost,
            sticky_bypass=self.allow_bypass,
        )

    # --- the Controller interface -----------------------------------------

    def choose(self, state: ControllerState) -> int:
        """Pick an action. Bypass is sticky for the rest of the request."""
        self._decisions += 1
        if self._bypassed:
            self._bypass_decisions += 1
            self.last_decision = ControllerDecision(
                action=0,
                reason="sticky bypass: drafting was retired earlier in this request",
                estimates=(),
                target_only_cost_ns=self.block_cost_ns(0, 1.0),
                sticky_bypass=True,
            )
            return 0
        decision = self.estimate(state)
        self.last_decision = decision
        if decision.action == 0:
            self._bypass_decisions += 1
            if decision.sticky_bypass:
                # Sticky because reactivating would need a draft cache rebuild
                # whose cost is not in this model (SPEC.md section 4.3).
                self._bypassed = True
        return decision.action

    def observe(self, block: BlockObservation) -> None:
        """Fold one completed block into the request-local estimates."""
        if block.action > 0:
            self._record_acceptance(block)
            if block.total_ns > 0:
                previous = self._observed_block_ns.get(block.action)
                alpha = self.profile.ewma_alpha
                self._observed_block_ns[block.action] = (
                    float(block.total_ns)
                    if previous is None
                    else alpha * block.total_ns + (1 - alpha) * previous
                )

    def _record_acceptance(self, block: BlockObservation) -> None:
        """Count successes and the one observed failure. Censor the rest.

        Positions after the first rejection were never evaluated, so they are
        recorded as neither success nor trial.
        """
        while len(self._trials) < block.proposed:
            self._successes.append(0)
            self._trials.append(0)
        for position in range(block.accepted):
            self._successes[position] += 1
            self._trials[position] += 1
        if block.rejection_position is not None:
            self._trials[block.rejection_position] += 1
