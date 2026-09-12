"""Force every accept/reject path with scripted adapters.

Real models reach these paths only by luck. A scripted adapter makes each one
reachable on demand, which is the only way to test rejection at position g,
accepted EOS, rejected EOS, and the all-accepted catch-up deterministically.

The scripted adapter is a real causal model in the one way that matters here:
the logits it returns at row ``i`` depend on the prefix ending at the token fed
at position ``i``. An adapter that ignored the prefix could not detect a
verification off-by-one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
import torch

from switchback.cache import CacheHandle
from switchback.decoder import EngineOptions, decode_speculative_greedy, decode_target_only
from switchback.events import FakeClock, ListSink
from switchback.types import ConfigError, DecodeConfig

VOCAB = 8
EOS = 7
PROMPT = [1, 2, 3]


@dataclass
class ScriptedAdapter:
    """A deterministic causal model defined by a prefix -> next-token function."""

    next_token: Callable[[tuple[int, ...]], int]
    name: str = "scripted"
    vocab_size: int = VOCAB
    checks: bool = True
    calls: list[tuple[int, ...]] = field(default_factory=list)
    widths: list[int] = field(default_factory=list)

    def new_cache(self) -> CacheHandle:
        return CacheHandle(name=self.name, backend=None, tokens=[], checks=self.checks)

    def backend_length(self, cache: CacheHandle) -> int | None:
        return None

    def forward(self, ids: torch.Tensor, cache: CacheHandle) -> torch.Tensor:
        fed = [int(value) for value in ids[0].tolist()]
        self.calls.append(tuple(fed))
        self.widths.append(len(fed))
        prefix = tuple(cache.tokens)
        rows = torch.full((1, len(fed), self.vocab_size), -10.0)
        for index, token in enumerate(fed):
            prefix = (*prefix, token)
            rows[0, index, self.next_token(prefix)] = 10.0
        cache.extend(fed)
        return rows

    def crop(self, cache: CacheHandle, length: int) -> None:
        cache.crop_to(length)

    def cache_length(self, cache: CacheHandle) -> int:
        return cache.length


def cycling(start: int) -> Callable[[tuple[int, ...]], int]:
    """Next token depends on the prefix length, so every position is distinct."""

    def choose(prefix: tuple[int, ...]) -> int:
        return (start + len(prefix)) % (VOCAB - 1)

    return choose


def agreeing() -> tuple[ScriptedAdapter, ScriptedAdapter]:
    """Target and draft that always agree: exercises the all-accepted path."""
    rule = cycling(0)
    return ScriptedAdapter(rule, "target"), ScriptedAdapter(rule, "draft")


def disagree_after(position: int) -> tuple[ScriptedAdapter, ScriptedAdapter]:
    """Draft matches the target until the block's ``position``-th candidate.

    ``position`` counts committed tokens from the start of the request, so a
    test can place the first disagreement at a chosen offset inside a block.
    """
    target_rule = cycling(0)

    def draft_rule(prefix: tuple[int, ...]) -> int:
        if len(prefix) - len(PROMPT) - 1 == position:
            return (target_rule(prefix) + 1) % (VOCAB - 1)
        return target_rule(prefix)

    return ScriptedAdapter(target_rule, "target"), ScriptedAdapter(draft_rule, "draft")


def always_disagreeing() -> tuple[ScriptedAdapter, ScriptedAdapter]:
    """The draft is wrong at every candidate, so no block is ever fully accepted."""
    target_rule = cycling(0)

    def draft_rule(prefix: tuple[int, ...]) -> int:
        return (target_rule(prefix) + 1) % (VOCAB - 1)

    return ScriptedAdapter(target_rule, "target"), ScriptedAdapter(draft_rule, "draft")


def greedy(**overrides: object) -> DecodeConfig:
    base: dict[str, object] = {
        "mode": "greedy",
        "temperature": 1.0,
        "max_new_tokens": 12,
        "eos_policy": "suppress_until_budget",
        "seed": 42,
    }
    base.update(overrides)
    return DecodeConfig(**base)  # type: ignore[arg-type]


def run(target, draft, gamma, config=None, eos=(EOS,), sink=None):
    return decode_speculative_greedy(
        target,
        draft,
        PROMPT,
        config or greedy(),
        list(eos),
        gamma=gamma,
        sink=sink,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True),
    )


# --- conformance -----------------------------------------------------------


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_speculation_matches_target_only_when_the_draft_always_agrees(gamma: int) -> None:
    target, draft = agreeing()
    reference = decode_target_only(
        ScriptedAdapter(cycling(0), "target"), PROMPT, greedy(), [EOS], clock=FakeClock()
    )
    result = run(target, draft, gamma)
    assert result.output_ids == reference.output_ids
    assert result.accepted == result.proposed


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
@pytest.mark.parametrize("position", [0, 1, 2, 3, 5])
def test_speculation_matches_target_only_wherever_the_draft_goes_wrong(
    gamma: int, position: int
) -> None:
    target, draft = disagree_after(position)
    reference = decode_target_only(
        ScriptedAdapter(cycling(0), "target"), PROMPT, greedy(), [EOS], clock=FakeClock()
    )
    result = run(target, draft, gamma)
    assert result.output_ids == reference.output_ids


# --- one forward call of width g+1 ----------------------------------------


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_verification_is_a_single_forward_call_of_width_gamma_plus_one(gamma: int) -> None:
    """One block of full width needs a budget of gamma + 2.

    The prefill emits one token, and the block length is capped at
    ``remaining - 1`` so a block can never overshoot the budget. So
    ``budget = 1 + (gamma + 1)`` is the smallest budget that admits a
    full-width block, and it is consumed exactly.
    """
    target, draft = agreeing()
    result = run(target, draft, gamma, config=greedy(max_new_tokens=gamma + 2))
    # The prefill, then exactly one verification call of width gamma + 1.
    assert target.widths == [len(PROMPT), gamma + 1]
    assert result.target_calls == 2
    assert len(result.output_ids) == gamma + 2


def test_every_block_uses_exactly_one_verification_call_of_its_own_width() -> None:
    """Holds for rejected blocks too, and for blocks the budget shortened."""
    target, draft = disagree_after(1)
    sink = ListSink()
    run(target, draft, 4, config=greedy(max_new_tokens=12), sink=sink)
    expected = [len(PROMPT)]
    for event in sink.events[1:]:
        expected.append(event.gamma + 1 if event.action == "speculative" else 1)
    assert target.widths == expected


# --- rejection at every position ------------------------------------------


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_rejection_lands_at_the_intended_position(position: int) -> None:
    gamma = 4
    target, draft = disagree_after(position)
    sink = ListSink()
    run(target, draft, gamma, config=greedy(max_new_tokens=gamma + 2), sink=sink)
    block = next(event for event in sink.events if event.action == "speculative")
    assert block.rejection_position == position
    assert block.accepted == position
    assert block.committed == position + 1


def test_full_acceptance_commits_gamma_plus_one_tokens() -> None:
    gamma = 4
    target, draft = agreeing()
    sink = ListSink()
    run(target, draft, gamma, config=greedy(max_new_tokens=gamma + 2), sink=sink)
    block = next(event for event in sink.events if event.action == "speculative")
    assert block.rejection_position is None
    assert block.accepted == gamma
    assert block.committed == gamma + 1


def test_all_accepted_triggers_exactly_one_draft_catch_up_call() -> None:
    """SPEC.md 4.3 step 4: the draft is one behind and the extra call is counted."""
    gamma = 4
    target, draft = agreeing()
    result = run(target, draft, gamma, config=greedy(max_new_tokens=gamma + 2))
    # prefill + gamma proposals + one catch-up.
    assert result.draft_calls == 1 + gamma + 1
    assert draft.widths == [len(PROMPT), *[1] * (gamma + 1)]


def test_a_rejected_block_never_pays_for_catch_up() -> None:
    """With the draft always wrong at the first candidate, no block ever
    reaches the all-accepted branch, so the only draft calls are the prefill
    and the proposals themselves."""
    gamma = 4
    target, draft = always_disagreeing()
    sink = ListSink()
    result = run(target, draft, gamma, config=greedy(max_new_tokens=12), sink=sink)
    proposals = sum(event.proposed for event in sink.events)
    assert all(event.accepted == 0 for event in sink.events)
    assert result.draft_calls == 1 + proposals


# --- cache invariants ------------------------------------------------------


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
@pytest.mark.parametrize("position", [0, 1, 3])
def test_caches_never_hold_a_rejected_suffix_at_a_boundary(gamma: int, position: int) -> None:
    """Invariants I2 and I4, read off the trace rather than asserted internally."""
    target, draft = disagree_after(position)
    sink = ListSink()
    result = run(target, draft, gamma, sink=sink)
    committed = 0
    for event in sink.events:
        committed += event.committed
        expected = len(PROMPT) + committed - 1
        assert event.target_cache_after == expected, event
        if event.action == "speculative":
            assert event.draft_cache_after == expected, event
    assert committed == len(result.output_ids)


def test_the_cache_ledger_is_checked_against_the_committed_tokens() -> None:
    """A silently wrong crop must raise rather than produce fluent output."""

    class BadCrop(ScriptedAdapter):
        def crop(self, cache: CacheHandle, length: int) -> None:
            cache.crop_to(max(0, length - 1))

    target = BadCrop(cycling(0), "target")
    _, draft = disagree_after(0)
    with pytest.raises(Exception, match="S\\[:-1\\]"):
        run(target, draft, 4)


# --- EOS and budget --------------------------------------------------------


def test_an_accepted_eos_ends_the_request_and_nothing_follows() -> None:
    """Invariant I7 across a block that commits several tokens at once."""

    def target_rule(prefix: tuple[int, ...]) -> int:
        return EOS if len(prefix) == len(PROMPT) + 2 else cycling(0)(prefix)

    target = ScriptedAdapter(target_rule, "target")
    draft = ScriptedAdapter(target_rule, "draft")
    result = run(
        target, draft, 4, config=greedy(eos_policy="respect", max_new_tokens=12), eos=(EOS,)
    )
    assert result.termination == "eos"
    assert result.output_ids[-1] == EOS
    assert EOS not in result.output_ids[:-1]


def test_an_eos_in_a_rejected_suffix_cannot_end_the_request() -> None:
    """SPEC.md 4.3 step 5. The draft proposes EOS; the target disagrees."""

    def target_rule(prefix: tuple[int, ...]) -> int:
        return cycling(0)(prefix)

    def draft_rule(prefix: tuple[int, ...]) -> int:
        # Propose EOS immediately, every time. The target never agrees.
        return EOS

    target = ScriptedAdapter(target_rule, "target")
    draft = ScriptedAdapter(draft_rule, "draft")
    sink = ListSink()
    result = run(
        target,
        draft,
        4,
        config=greedy(eos_policy="respect", max_new_tokens=10),
        eos=(EOS,),
        sink=sink,
    )
    assert result.termination == "budget"
    assert len(result.output_ids) == 10
    assert EOS not in result.output_ids
    # A proposed stop token ends drafting early but is still verified, so each
    # speculative block proposes exactly one candidate and accepts none.
    speculative = [event for event in sink.events if event.action == "speculative"]
    assert speculative
    assert all(event.proposed == 1 and event.accepted == 0 for event in speculative)
    assert result.proposed == len(speculative)


def test_suppressed_eos_is_never_proposed_or_committed() -> None:
    def rule(prefix: tuple[int, ...]) -> int:
        return EOS

    target = ScriptedAdapter(rule, "target")
    draft = ScriptedAdapter(rule, "draft")
    result = run(target, draft, 4, config=greedy(max_new_tokens=9), eos=(EOS,))
    assert result.termination == "budget"
    assert EOS not in result.output_ids
    assert len(result.output_ids) == 9


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 9, 17])
@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_the_budget_is_never_exceeded_or_undershot(budget: int, gamma: int) -> None:
    target, draft = agreeing()
    result = run(target, draft, gamma, config=greedy(max_new_tokens=budget))
    assert len(result.output_ids) == budget
    assert len(result.token_release_ns) == budget


def test_a_single_token_budget_never_initializes_the_draft() -> None:
    """SPEC.md 4.3: a request that ends at its first token pays no draft prefill."""
    target, draft = agreeing()
    result = run(target, draft, 8, config=greedy(max_new_tokens=1))
    assert result.draft_calls == 0
    assert result.counters["draft_initialized"] == 0
    assert result.counters["draft_cache_length"] is None
    assert draft.calls == []


def test_the_last_remaining_token_uses_a_target_only_step() -> None:
    target, draft = agreeing()
    sink = ListSink()
    run(target, draft, 4, config=greedy(max_new_tokens=7), sink=sink)
    assert sink.events[-1].action in {"target_only", "speculative"}
    # A block emits at least two tokens, so a budget of 7 after the prefill token
    # leaves 6, which 4+1 then a final single cannot overshoot.
    assert sum(event.committed for event in sink.events) == 7


# --- configuration ---------------------------------------------------------


def test_sampled_mode_is_refused_by_the_greedy_engine() -> None:
    target, draft = agreeing()
    with pytest.raises(ConfigError, match="greedy mode"):
        run(target, draft, 4, config=DecodeConfig("sample", 0.7, 8, "respect", 1))


def test_gamma_must_be_positive() -> None:
    target, draft = agreeing()
    with pytest.raises(ConfigError, match="gamma >= 1"):
        run(target, draft, 0)


def test_counters_report_proposals_and_acceptances() -> None:
    target, draft = agreeing()
    result = run(target, draft, 4, config=greedy(max_new_tokens=11))
    assert result.proposed > 0
    assert result.accepted == result.proposed
    assert result.counters["gamma"] == 4
    assert result.counters["draft_initialized"] == 1


def test_unused_proposals_count_towards_the_denominator() -> None:
    """Acceptance includes the suffix thrown away after a rejection.

    The first block proposes its full width and accepts nothing, so all four
    candidates land in the denominator (SPEC.md section 9.5).
    """
    gamma = 4
    target, draft = disagree_after(0)
    sink = ListSink()
    result = run(target, draft, gamma, config=greedy(max_new_tokens=gamma + 2), sink=sink)
    first = next(event for event in sink.events if event.action == "speculative")
    assert first.proposed == gamma
    assert first.accepted == 0
    assert first.committed == 1
    assert result.accepted < result.proposed
