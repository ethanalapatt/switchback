"""Fixed greedy speculation on real tiny transformers, FP32 on CPU.

The scripted-adapter tests force every control-flow path; these check that the
same code is right when the rows come from an actual model with an actual KV
cache. Both are needed: a scripted adapter cannot catch a rotary-position bug,
and a real model cannot be made to reject at a chosen position on demand.
"""

from __future__ import annotations

import pytest
import torch

from switchback.decoder import (
    EngineOptions,
    decode_speculative_greedy,
    decode_target_only,
)
from switchback.events import FakeClock, ListSink
from switchback.models.qwen import QwenAdapter
from switchback.models.tiny import EOS_ID, build_tiny_model
from switchback.types import DecodeConfig

PROMPT = [4, 20, 36, 7, 50, 9]
LONGER_PROMPT = [4, 20, 36, 7, 50, 9, 11, 30, 44, 8, 60, 15, 22, 70, 33, 41]
GAMMAS = (1, 2, 4, 8)


@pytest.fixture(scope="module")
def models():
    target = build_tiny_model(seed=1234, hidden_size=64, layers=2)
    draft = build_tiny_model(seed=5678, hidden_size=32, layers=1)
    return target, draft


@pytest.fixture
def adapters(models):
    target, draft = models
    return (
        QwenAdapter(
            model=target.model, device="cpu", vocab_size=target.logits_vocab_size, name="target"
        ),
        QwenAdapter(
            model=draft.model, device="cpu", vocab_size=draft.logits_vocab_size, name="draft"
        ),
    )


def greedy(**overrides) -> DecodeConfig:
    base = dict(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=24,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    base.update(overrides)
    return DecodeConfig(**base)


def options() -> EngineOptions:
    return EngineOptions(correctness_checks=True, device="cpu")


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("prompt", [PROMPT, LONGER_PROMPT], ids=["short", "long"])
def test_fixed_speculation_equals_target_only(adapters, gamma, prompt) -> None:
    """The milestone 4 gate: exact token-ID agreement for every fixed length."""
    target, draft = adapters
    config = greedy()
    reference = decode_target_only(target, prompt, config, [EOS_ID], clock=FakeClock())
    result = decode_speculative_greedy(
        target,
        draft,
        prompt,
        config,
        [EOS_ID],
        gamma=gamma,
        clock=FakeClock(),
        options=options(),
    )
    assert result.output_ids == reference.output_ids


@pytest.mark.parametrize("gamma", GAMMAS)
def test_a_draft_identical_to_the_target_accepts_everything(models, gamma) -> None:
    """With draft == target every candidate is accepted, which exercises the
    all-accepted branch and its catch-up call on a real model rather than a
    scripted one."""
    target_model, _ = models
    target = QwenAdapter(
        model=target_model.model,
        device="cpu",
        vocab_size=target_model.logits_vocab_size,
        name="target",
    )
    draft = QwenAdapter(
        model=target_model.model,
        device="cpu",
        vocab_size=target_model.logits_vocab_size,
        name="draft",
    )
    config = greedy(max_new_tokens=17)
    reference = decode_target_only(target, PROMPT, config, [EOS_ID], clock=FakeClock())
    sink = ListSink()
    result = decode_speculative_greedy(
        target,
        draft,
        PROMPT,
        config,
        [EOS_ID],
        gamma=gamma,
        sink=sink,
        clock=FakeClock(),
        options=options(),
    )
    assert result.output_ids == reference.output_ids
    assert result.accepted == result.proposed > 0
    speculative = [event for event in sink.events if event.action == "speculative"]
    assert all(event.rejection_position is None for event in speculative)
    # Far fewer target calls than tokens: that is the entire point of drafting.
    assert result.target_calls < len(result.output_ids)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_cache_boundaries_hold_on_a_real_model(adapters, gamma) -> None:
    target, draft = adapters
    sink = ListSink()
    result = decode_speculative_greedy(
        target,
        draft,
        PROMPT,
        greedy(max_new_tokens=20),
        [EOS_ID],
        gamma=gamma,
        sink=sink,
        clock=FakeClock(),
        options=options(),
    )
    committed = 0
    for event in sink.events:
        committed += event.committed
        assert event.target_cache_after == len(PROMPT) + committed - 1
        if event.action == "speculative":
            assert event.draft_cache_after == len(PROMPT) + committed - 1
    assert committed == len(result.output_ids)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_eos_terminates_identically_to_target_only(adapters, gamma) -> None:
    target, draft = adapters
    # Make a common token a stop token so termination actually happens.
    config = greedy(eos_policy="respect", max_new_tokens=24)
    stops = [EOS_ID, 62, 35]
    reference = decode_target_only(target, PROMPT, config, stops, clock=FakeClock())
    result = decode_speculative_greedy(
        target,
        draft,
        PROMPT,
        config,
        stops,
        gamma=gamma,
        clock=FakeClock(),
        options=options(),
    )
    assert result.output_ids == reference.output_ids
    assert result.termination == reference.termination
    if result.termination == "eos":
        assert result.output_ids[-1] in stops
        assert not set(result.output_ids[:-1]) & set(stops)


@pytest.mark.parametrize("budget", [1, 2, 3, 6, 11, 23])
@pytest.mark.parametrize("gamma", GAMMAS)
def test_budget_is_met_exactly_for_every_length(adapters, budget, gamma) -> None:
    target, draft = adapters
    result = decode_speculative_greedy(
        target,
        draft,
        PROMPT,
        greedy(max_new_tokens=budget),
        [EOS_ID],
        gamma=gamma,
        clock=FakeClock(),
        options=options(),
    )
    assert len(result.output_ids) == budget


def test_requests_are_independent(adapters) -> None:
    target, draft = adapters
    config = greedy(max_new_tokens=12)
    first = decode_speculative_greedy(
        target, draft, PROMPT, config, [EOS_ID], gamma=4, clock=FakeClock(), options=options()
    )
    decode_speculative_greedy(
        target,
        draft,
        LONGER_PROMPT,
        config,
        [EOS_ID],
        gamma=4,
        clock=FakeClock(),
        options=options(),
    )
    again = decode_speculative_greedy(
        target, draft, PROMPT, config, [EOS_ID], gamma=4, clock=FakeClock(), options=options()
    )
    assert first.output_ids == again.output_ids


def test_correctness_checks_do_not_change_the_output(adapters) -> None:
    target, draft = adapters
    config = greedy(max_new_tokens=16)
    checked = decode_speculative_greedy(
        target,
        draft,
        PROMPT,
        config,
        [EOS_ID],
        gamma=4,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True, device="cpu"),
    )
    unchecked = decode_speculative_greedy(
        target,
        draft,
        PROMPT,
        config,
        [EOS_ID],
        gamma=4,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=False, device="cpu"),
    )
    assert checked.output_ids == unchecked.output_ids
    assert checked.target_calls == unchecked.target_calls
    assert checked.draft_calls == unchecked.draft_calls


def test_target_calls_scale_with_acceptance_not_with_tokens(adapters) -> None:
    """A sanity check on the counters that the cost model will depend on."""
    target, draft = adapters
    config = greedy(max_new_tokens=24)
    baseline = decode_target_only(target, PROMPT, config, [EOS_ID], clock=FakeClock())
    assert baseline.target_calls == 24
    for gamma in GAMMAS:
        result = decode_speculative_greedy(
            target,
            draft,
            PROMPT,
            config,
            [EOS_ID],
            gamma=gamma,
            clock=FakeClock(),
            options=options(),
        )
        # Independently seeded tiny models rarely agree, so speculation here
        # costs roughly one target call per token plus the draft work. That is a
        # real regime, and the benchmark has to report it rather than hide it.
        assert result.target_calls <= 24
        assert result.draft_calls >= result.proposed


def test_logits_row_alignment_is_load_bearing(adapters) -> None:
    """Verifying against the wrong row must break the output, or the equality
    test above proves nothing."""
    from switchback.decoder import _verify_greedy

    rows = torch.full((1, 4, 16), -5.0)
    proposals = [3, 5, 7]
    for index, token in enumerate(proposals):
        rows[0, index, token] = 5.0
    rows[0, 3, 11] = 5.0
    correct = _verify_greedy(rows, proposals, ())
    assert correct.all_accepted
    assert correct.emitted == (3, 5, 7, 11)

    shifted = rows.roll(1, dims=1)
    wrong = _verify_greedy(shifted, proposals, ())
    assert not wrong.all_accepted


def test_verification_row_count_is_checked(adapters) -> None:
    from switchback.decoder import _verify_greedy

    with pytest.raises(AssertionError, match="4 rows for 3 proposals"):
        _verify_greedy(torch.zeros((1, 3, 8)), [1, 2, 3], ())
