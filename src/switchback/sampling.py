"""Numerical sampling: probability transforms, verification, and correction.

This module is pure. It never reads a clock, a device counter, a controller
state, or a benchmark label (SPEC.md section 3), and it holds no cache or model
state. Every stochastic decision is delegated to a :class:`RandomSource` so the
algorithm can be driven by a real RNG in production and enumerated exhaustively
in tests.

Tensor axis names: ``V`` vocabulary, ``T`` positions in one forward call.

Two design choices make exhaustive enumeration exact rather than approximate:

* Acceptance is expressed as a ratio, ``accept_ratio(p[d], q[d])``, not as a
  precomputed probability. ``min(1, p/q)`` of two dyadic rationals is generally
  not dyadic (``(1/4)/(3/4) = 1/3``), so rounding it to FP32 before the draw
  would put an irreducible 1e-8 error into every enumerated path.
* Categorical draws take **unnormalized non-negative weights**. The residual
  ``max(p - q, 0)`` is exact for dyadic inputs; dividing it by its sum is not.
  A production source normalizes in FP32 to sample; an enumerating source
  normalizes exactly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from switchback.types import DecodeConfig, NumericalSamplingError

# The three independent randomness streams of one request (SPEC.md section 4.1).
PROPOSAL_STREAM = "proposal"
ACCEPTANCE_STREAM = "acceptance"
CORRECTION_STREAM = "correction"
STREAMS: tuple[str, ...] = (PROPOSAL_STREAM, ACCEPTANCE_STREAM, CORRECTION_STREAM)

# Sampling is done in FP32 regardless of the model's compute dtype.
PROBABILITY_DTYPE = "float32"


def derive_stream_seed(request_seed: int, stream: str) -> int:
    """Derive one stream's seed from the request seed.

    Recipe: ``SHA-256("switchback/" + request_seed + "/" + stream)``, first eight
    bytes big-endian, masked to 63 bits. Documented rather than clever so a
    reviewer can recompute any stream's seed by hand, and so two streams cannot
    accidentally coincide the way ``seed + 1`` schemes do.
    """
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")
    payload = f"switchback/{request_seed}/{stream}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class RandomSource(Protocol):
    """Every stochastic decision in this module goes through this interface."""

    def categorical(self, weights: Any, stream: str) -> int:
        """Draw an index with probability proportional to non-negative ``weights`` ``[V]``."""

    def accept_ratio(self, numerator: float, denominator: float, stream: str) -> bool:
        """Return True with probability ``min(1, numerator / denominator)``."""


@dataclass
class TorchRandomSource:
    """Production source: three seeded ``torch.Generator`` streams.

    Categorical draws use inverse-CDF sampling on one uniform, so the number of
    random values consumed per decision is exactly one and is independent of the
    vocabulary size. That keeps a replayed request aligned with the original even
    if an unrelated part of the engine changes.
    """

    request_seed: int
    device: str = "cpu"
    _generators: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        import torch

        for stream in STREAMS:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(derive_stream_seed(self.request_seed, stream))
            self._generators[stream] = generator

    def _uniform(self, stream: str) -> float:
        import torch

        generator = self._generators[stream]
        value = torch.rand((), generator=generator, device=self.device, dtype=torch.float32)
        return float(value)

    def categorical(self, weights: Any, stream: str) -> int:
        import torch

        probabilities = normalize(weights)
        cumulative = torch.cumsum(probabilities.double(), dim=-1)
        draw = self._uniform(stream) * float(cumulative[-1])
        index = int(torch.searchsorted(cumulative, torch.tensor(draw, dtype=cumulative.dtype)))
        index = min(index, int(probabilities.shape[-1]) - 1)
        if float(probabilities[index]) > 0.0:
            return index
        # Only reachable through floating-point drift at a cumulative boundary.
        # Move to the nearest token that actually carries mass rather than
        # emitting a zero-probability token (invariant I8).
        support = torch.nonzero(probabilities > 0, as_tuple=False).flatten()
        if support.numel() == 0:
            raise NumericalSamplingError("categorical draw from an all-zero distribution")
        later = support[support >= index]
        return int(later[0]) if later.numel() else int(support[-1])

    def accept_ratio(self, numerator: float, denominator: float, stream: str) -> bool:
        if denominator <= 0.0:
            raise NumericalSamplingError(
                f"acceptance ratio has non-positive denominator {denominator!r}; "
                f"the candidate had no support under the distribution that drew it"
            )
        # u * q <= p avoids forming the quotient, which keeps the comparison
        # exact for dyadic p and q.
        return self._uniform(stream) * denominator <= numerator


def validate_distribution(probabilities: Any, name: str) -> None:
    """Reject a distribution that is nonfinite, negative, or has no support."""
    import torch

    if not torch.isfinite(probabilities).all():
        raise NumericalSamplingError(f"{name} contains a nonfinite value")
    if bool((probabilities < 0).any()):
        raise NumericalSamplingError(f"{name} contains a negative probability")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise NumericalSamplingError(f"{name} has zero total mass")


def normalize(weights: Any) -> Any:
    """Scale non-negative ``weights`` ``[V]`` to sum to one, in FP32."""
    import torch

    values = weights.to(torch.float32)
    validate_distribution(values, "weights")
    return values / values.sum()


def softmax_probabilities(
    logits: Any,
    temperature: float,
    forbidden_ids: Sequence[int] = (),
) -> Any:
    """Turn logits ``[V]`` into a probability vector ``[V]`` in FP32.

    Order matters and is fixed here: forbidden tokens are masked to ``-inf``
    *before* the temperature division and the softmax, so suppressing EOS
    renormalizes over the remaining tokens instead of leaving a hole. No top-k,
    top-p, or penalty transform exists in the MVP, by construction rather than
    by configuration (SPEC.md section 4.1).
    """
    import torch

    if not (temperature > 0.0) or temperature == float("inf"):
        raise NumericalSamplingError(
            f"temperature must be finite and positive, got {temperature!r}"
        )
    values = logits.to(torch.float32).clone()
    if not torch.isfinite(values).all():
        raise NumericalSamplingError("logits contain a nonfinite value")
    if forbidden_ids:
        index = torch.tensor(list(forbidden_ids), dtype=torch.long, device=values.device)
        values.index_fill_(-1, index, float("-inf"))
    if not bool(torch.isfinite(values).any()):
        raise NumericalSamplingError("every token was suppressed; no support remains")
    values = values / temperature
    # Stable softmax: subtract the maximum before exponentiating.
    values = values - values.max()
    probabilities = torch.exp(values)
    total = probabilities.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        raise NumericalSamplingError("softmax underflowed to zero total mass")
    probabilities = probabilities / total
    validate_distribution(probabilities, "softmax probabilities")
    return probabilities


def greedy_token(logits: Any, forbidden_ids: Sequence[int] = ()) -> int:
    """Argmax over logits ``[V]`` with smallest-token-ID tie breaking.

    ``torch.argmax`` does not promise which index wins a tie, so the tie break
    is made explicit: greedy decoding must be bit-for-bit reproducible across
    engines for the conformance check in invariant I6.
    """
    import torch

    values = logits.to(torch.float32).clone()
    if not torch.isfinite(values).all():
        raise NumericalSamplingError("logits contain a nonfinite value")
    if forbidden_ids:
        index = torch.tensor(list(forbidden_ids), dtype=torch.long, device=values.device)
        values.index_fill_(-1, index, float("-inf"))
    if not bool(torch.isfinite(values).any()):
        raise NumericalSamplingError("every token was suppressed; no support remains")
    maximum = values.max()
    winners = torch.nonzero(values == maximum, as_tuple=False).flatten()
    return int(winners[0])


def positive_residual(target: Any, draft: Any) -> Any:
    """Unnormalized correction weights ``max(p - q, 0)`` over ``[V]``.

    Returned unnormalized on purpose: the difference of two dyadic rationals is
    exact, its normalization is not, and the caller's random source decides how
    to divide. Subtraction roundoff is clamped at zero before any use.
    """
    import torch

    p = target.to(torch.float32)
    q = draft.to(torch.float32)
    if p.shape != q.shape:
        raise NumericalSamplingError(
            f"target {tuple(p.shape)} and draft {tuple(q.shape)} distributions differ in shape"
        )
    validate_distribution(p, "target distribution")
    validate_distribution(q, "draft distribution")
    return torch.clamp(p - q, min=0.0)


def draw_correction(target: Any, draft: Any, source: RandomSource) -> int:
    """Draw one correction token from the normalized positive residual.

    A zero-mass residual after a rejection is a numerical failure, not a licence
    to substitute another distribution: in exact arithmetic a rejection can only
    happen when some token has ``p(x) > q(x)`` (SPEC.md section 4.1).
    """
    weights = positive_residual(target, draft)
    if float(weights.sum()) <= 0.0:
        raise NumericalSamplingError(
            "residual max(p - q, 0) has zero mass after a rejection; "
            "the run is invalid and must not be reported"
        )
    return source.categorical(weights, CORRECTION_STREAM)


@dataclass(frozen=True)
class BlockOutcome:
    """Result of verifying one speculative block. Only committed tokens appear.

    ``emitted`` always contains ``accepted + 1`` tokens: the accepted candidates
    followed by either a correction (on rejection) or a bonus (on full
    acceptance). ``rejection_position`` is the zero-based index of the first
    rejected candidate, or ``None`` when every candidate was accepted.
    """

    proposed: tuple[int, ...]
    accepted: int
    rejection_position: int | None
    emitted: tuple[int, ...]
    all_accepted: bool

    def __post_init__(self) -> None:
        if len(self.emitted) != self.accepted + 1:
            raise AssertionError(
                f"block emitted {len(self.emitted)} tokens for {self.accepted} accepted"
            )
        if self.all_accepted != (self.rejection_position is None):
            raise AssertionError("all_accepted disagrees with rejection_position")
        if self.rejection_position is not None and self.rejection_position != self.accepted:
            raise AssertionError("rejection position must equal the accepted count")


RowFn = Callable[[tuple[int, ...]], Any]
"""Maps a committed prefix to that position's next-token distribution ``[V]``."""


def speculative_block(
    prefix: tuple[int, ...],
    gamma: int,
    target_row: RowFn,
    draft_row: RowFn,
    source: RandomSource,
) -> BlockOutcome:
    """Propose ``gamma`` tokens, verify them, and commit the accepted prefix.

    ``target_row``/``draft_row`` hide where the distributions come from, so this
    function is identical whether it is driven by two real models or by a tiny
    enumerable tree. It returns only committed tokens; a rejected candidate and
    everything after it are discarded (invariants I1 and I4).

    The target is queried at ``gamma + 1`` prefixes, not ``gamma``: row ``j``
    predicts candidate ``j``, and row ``gamma`` supplies the bonus token.
    """
    if gamma < 1:
        raise ValueError(f"speculative_block requires gamma >= 1, got {gamma}")

    proposals: list[int] = []
    draft_rows: list[Any] = []
    current = prefix
    for _ in range(gamma):
        q = draft_row(current)
        validate_distribution(q, "draft distribution")
        candidate = source.categorical(q, PROPOSAL_STREAM)
        if float(q[candidate]) <= 0.0:
            raise NumericalSamplingError(
                "draft proposed a token with zero probability under its own distribution"
            )
        draft_rows.append(q)
        proposals.append(candidate)
        current = (*current, candidate)

    for index, candidate in enumerate(proposals):
        p = target_row(prefix + tuple(proposals[:index]))
        validate_distribution(p, "target distribution")
        q = draft_rows[index]
        if source.accept_ratio(float(p[candidate]), float(q[candidate]), ACCEPTANCE_STREAM):
            continue
        correction = draw_correction(p, q, source)
        return BlockOutcome(
            proposed=tuple(proposals),
            accepted=index,
            rejection_position=index,
            emitted=(*proposals[:index], correction),
            all_accepted=False,
        )

    bonus_row = target_row(prefix + tuple(proposals))
    validate_distribution(bonus_row, "target distribution")
    bonus = source.categorical(bonus_row, CORRECTION_STREAM)
    return BlockOutcome(
        proposed=tuple(proposals),
        accepted=gamma,
        rejection_position=None,
        emitted=(*proposals, bonus),
        all_accepted=True,
    )


def greedy_block(
    prefix: tuple[int, ...],
    gamma: int,
    target_row: RowFn,
    draft_row: RowFn,
) -> BlockOutcome:
    """Greedy speculation: argmax proposals verified against target argmax.

    Deterministic and separate from the sampled path. Tie breaking is shared
    with target-only decoding so the two produce identical token IDs.
    """
    if gamma < 1:
        raise ValueError(f"greedy_block requires gamma >= 1, got {gamma}")

    proposals: list[int] = []
    current = prefix
    for _ in range(gamma):
        candidate = _argmax_of(draft_row(current))
        proposals.append(candidate)
        current = (*current, candidate)

    for index, candidate in enumerate(proposals):
        expected = _argmax_of(target_row(prefix + tuple(proposals[:index])))
        if expected == candidate:
            continue
        return BlockOutcome(
            proposed=tuple(proposals),
            accepted=index,
            rejection_position=index,
            emitted=(*proposals[:index], expected),
            all_accepted=False,
        )

    bonus = _argmax_of(target_row(prefix + tuple(proposals)))
    return BlockOutcome(
        proposed=tuple(proposals),
        accepted=gamma,
        rejection_position=None,
        emitted=(*proposals, bonus),
        all_accepted=True,
    )


def _argmax_of(probabilities: Any) -> int:
    import torch

    validate_distribution(probabilities, "distribution")
    maximum = probabilities.max()
    winners = torch.nonzero(probabilities == maximum, as_tuple=False).flatten()
    return int(winners[0])


def forbidden_token_ids(config: DecodeConfig, eos_token_ids: Sequence[int]) -> tuple[int, ...]:
    """Token IDs masked out for this request under the configured EOS policy."""
    if config.eos_policy == "suppress_until_budget":
        return tuple(sorted(set(int(value) for value in eos_token_ids)))
    return ()


def target_only_step(prefix: tuple[int, ...], target_row: RowFn, source: RandomSource) -> int:
    """Draw one token from the target distribution at ``prefix``.

    Used for the bypass action and for the final token of a request. It consumes
    the correction stream, the same stream a bonus token uses, so that switching
    between speculation and bypass does not reorder the proposal stream.
    """
    row = target_row(prefix)
    validate_distribution(row, "target distribution")
    return source.categorical(row, CORRECTION_STREAM)


def speculative_sequence(
    prefix: tuple[int, ...],
    gamma: int,
    max_new_tokens: int,
    target_row: RowFn,
    draft_row: RowFn,
    source: RandomSource,
) -> tuple[int, ...]:
    """Emit exactly ``max_new_tokens`` tokens using repeated speculative blocks.

    The block length is capped at ``remaining - 1`` and a single remaining token
    is taken with a target-only step (SPEC.md section 4.3). Because a block emits
    at most ``g + 1`` tokens, that cap makes overshoot impossible, so the output
    is never truncated: truncating would condition on the future and break the
    distribution argument.

    Model-free by construction. The cache bookkeeping that accompanies this in a
    real engine lives in the decoder, not here.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    emitted: list[int] = []
    while len(emitted) < max_new_tokens:
        remaining = max_new_tokens - len(emitted)
        current = prefix + tuple(emitted)
        if remaining == 1:
            emitted.append(target_only_step(current, target_row, source))
            continue
        outcome = speculative_block(
            current, min(gamma, remaining - 1), target_row, draft_row, source
        )
        emitted.extend(outcome.emitted)
    return tuple(emitted)


def greedy_sequence(
    prefix: tuple[int, ...],
    gamma: int,
    max_new_tokens: int,
    target_row: RowFn,
    draft_row: RowFn,
) -> tuple[int, ...]:
    """Greedy counterpart of :func:`speculative_sequence`. Deterministic."""
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    emitted: list[int] = []
    while len(emitted) < max_new_tokens:
        remaining = max_new_tokens - len(emitted)
        current = prefix + tuple(emitted)
        if remaining == 1 or gamma < 1:
            emitted.append(_argmax_of(target_row(current)))
            continue
        outcome = greedy_block(current, min(gamma, remaining - 1), target_row, draft_row)
        emitted.extend(outcome.emitted)
    return tuple(emitted)


def greedy_target_sequence(
    prefix: tuple[int, ...], max_new_tokens: int, target_row: RowFn
) -> tuple[int, ...]:
    """Target-only greedy decoding: the reference for greedy conformance."""
    emitted: list[int] = []
    for _ in range(max_new_tokens):
        emitted.append(_argmax_of(target_row(prefix + tuple(emitted))))
    return tuple(emitted)
