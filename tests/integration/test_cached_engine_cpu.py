"""Milestone 3 on CPU: the cache is validated before any speculation exists.

FP32 on the tiny fixture first, as SPEC.md M3 requires. The BF16 GPU checks live
in tests/integration/test_real_models_gpu.py. Tolerances here are documented
constants, never widened to make a comparison pass.
"""

from __future__ import annotations

import pytest
import torch

from switchback.cache import CacheError
from switchback.decoder import EngineOptions, decode_target_only, select_token
from switchback.events import FakeClock, ListSink
from switchback.models.qwen import QwenAdapter
from switchback.models.tiny import EOS_ID, TINY_VOCAB_SIZE, build_tiny_model
from switchback.sampling import TorchRandomSource
from switchback.types import MAX_CONTEXT_TOKENS, ConfigError, DecodeConfig

# FP32 eager attention over a 6-layer-equivalent tiny model. Logits reach ~15 in
# magnitude, so 1e-4 is roughly 1e-5 relative: loose enough for a different
# reduction order, tight enough that a one-position cache offset fails by orders
# of magnitude.
LOGIT_ATOL = 1e-4
LOGIT_RTOL = 1e-4

PROMPT = [4, 20, 36, 7, 50, 9]


@pytest.fixture(scope="module")
def fixture_model():
    return build_tiny_model(seed=1234)


@pytest.fixture
def adapter(fixture_model):
    return QwenAdapter(
        model=fixture_model.model,
        device="cpu",
        vocab_size=fixture_model.logits_vocab_size,
        name="target",
    )


def greedy(**overrides) -> DecodeConfig:
    base = dict(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=12,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    base.update(overrides)
    return DecodeConfig(**base)


def full_prefix_logits(model, ids: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        return model(input_ids=torch.tensor([ids])).logits


# --- cache / no-cache differential ----------------------------------------


@pytest.mark.parametrize("split", [1, 2, 5])
def test_cached_logits_match_full_prefix_recomputation(fixture_model, adapter, split) -> None:
    ids = [*PROMPT, 11, 30, 44]
    reference = full_prefix_logits(fixture_model.model, ids)
    cache = adapter.new_cache()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(ids), split):
            chunk = torch.tensor([ids[start : start + split]])
            pieces.append(adapter.forward(chunk, cache))
    cached = torch.cat(pieces, dim=1)
    assert cached.shape == reference.shape
    assert torch.allclose(cached, reference, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
    # Predictions must agree exactly, which is the property decoding depends on.
    assert torch.equal(cached.argmax(-1), reference.argmax(-1))


def test_logits_row_i_predicts_the_token_after_position_i(fixture_model, adapter) -> None:
    """Invariant I3, the alignment speculative verification relies on."""
    ids = [*PROMPT, 11, 30]
    reference = full_prefix_logits(fixture_model.model, ids)
    for cut in range(1, len(ids)):
        partial = full_prefix_logits(fixture_model.model, ids[:cut])
        assert torch.allclose(
            partial[0, -1], reference[0, cut - 1], atol=LOGIT_ATOL, rtol=LOGIT_RTOL
        )


def test_cropping_then_replaying_reproduces_the_uncropped_logits(adapter) -> None:
    """A rollback must leave the cache exactly as if the suffix never happened."""
    cache = adapter.new_cache()
    with torch.inference_mode():
        adapter.forward(torch.tensor([PROMPT]), cache)
        expected = adapter.forward(torch.tensor([[11, 30]]), cache)
        adapter.crop(cache, len(PROMPT))
        # Poison the cache with a wrong suffix, then roll it back.
        adapter.forward(torch.tensor([[77, 88, 91]]), cache)
        adapter.crop(cache, len(PROMPT))
        replayed = adapter.forward(torch.tensor([[11, 30]]), cache)
    assert torch.equal(replayed, expected)
    assert cache.tokens == [*PROMPT, 11, 30]


def test_a_stale_suffix_actually_changes_the_logits(adapter) -> None:
    """Without this, the rollback test above could pass on a no-op cache."""
    clean = adapter.new_cache()
    dirty = adapter.new_cache()
    with torch.inference_mode():
        adapter.forward(torch.tensor([PROMPT]), clean)
        left = adapter.forward(torch.tensor([[11]]), clean)
        adapter.forward(torch.tensor([[*PROMPT, 77, 88]]), dirty)
        right = adapter.forward(torch.tensor([[11]]), dirty)
    assert not torch.allclose(left, right, atol=LOGIT_ATOL)


def test_cache_length_tracks_the_backend(adapter) -> None:
    cache = adapter.new_cache()
    assert adapter.cache_length(cache) == 0
    with torch.inference_mode():
        adapter.forward(torch.tensor([PROMPT]), cache)
    assert adapter.cache_length(cache) == len(PROMPT) == adapter.backend_length(cache)
    adapter.crop(cache, 3)
    assert adapter.cache_length(cache) == 3 == adapter.backend_length(cache)


def test_forward_rejects_bad_shapes_and_empty_input(adapter) -> None:
    cache = adapter.new_cache()
    with pytest.raises(ConfigError, match=r"\[1, T\]"):
        adapter.forward(torch.tensor([1, 2]), cache)
    with pytest.raises(ConfigError, match=r"\[1, T\]"):
        adapter.forward(torch.tensor([[1], [2]]), cache)
    with pytest.raises(ConfigError, match="at least one token"):
        adapter.forward(torch.zeros((1, 0), dtype=torch.long), cache)


def test_forward_refuses_to_exceed_the_context_bound(adapter) -> None:
    cache = adapter.new_cache()
    cache.tokens = list(range(MAX_CONTEXT_TOKENS - 1))
    with pytest.raises(ConfigError, match="context bound"):
        adapter.forward(torch.tensor([[1, 2, 3]]), cache)


def test_ledger_disagreement_with_the_backend_is_caught(adapter) -> None:
    cache = adapter.new_cache()
    with torch.inference_mode():
        adapter.forward(torch.tensor([PROMPT]), cache)
    cache.tokens.pop()
    with pytest.raises(CacheError, match="backend says"), torch.inference_mode():
        adapter.forward(torch.tensor([[11]]), cache)


# --- target-only engine ----------------------------------------------------


def reference_greedy(model, prompt: list[int], count: int, forbidden: set[int]) -> tuple[int, ...]:
    """Uncached greedy decoding: recompute the whole prefix every step."""
    ids = list(prompt)
    out: list[int] = []
    with torch.inference_mode():
        for _ in range(count):
            row = model(input_ids=torch.tensor([ids])).logits[0, -1].clone()
            for token in forbidden:
                row[token] = float("-inf")
            chosen = int(row.argmax())
            out.append(chosen)
            ids.append(chosen)
    return tuple(out)


def test_greedy_matches_uncached_recomputation(fixture_model, adapter) -> None:
    config = greedy(max_new_tokens=16)
    result = decode_target_only(adapter, PROMPT, config, [EOS_ID], clock=FakeClock())
    expected = reference_greedy(fixture_model.model, PROMPT, 16, {EOS_ID})
    assert result.output_ids == expected
    assert result.target_calls == 16


def test_prompt_of_length_one_works(fixture_model, adapter) -> None:
    result = decode_target_only(adapter, [7], greedy(max_new_tokens=5), [EOS_ID], clock=FakeClock())
    assert result.output_ids == reference_greedy(fixture_model.model, [7], 5, {EOS_ID})


def test_budget_of_one_emits_exactly_one_token(adapter) -> None:
    result = decode_target_only(
        adapter, PROMPT, greedy(max_new_tokens=1), [EOS_ID], clock=FakeClock()
    )
    assert len(result.output_ids) == 1
    assert result.first_token_ns == result.last_token_ns
    assert result.termination == "budget"


def test_output_never_exceeds_the_budget(adapter) -> None:
    for budget in (1, 2, 7, 13):
        result = decode_target_only(
            adapter, PROMPT, greedy(max_new_tokens=budget), [EOS_ID], clock=FakeClock()
        )
        assert len(result.output_ids) == budget


def test_prompt_plus_budget_beyond_the_context_bound_is_refused(adapter) -> None:
    with pytest.raises(ConfigError, match="context bound"):
        decode_target_only(
            adapter,
            list(range(MAX_CONTEXT_TOKENS - 4)),
            greedy(max_new_tokens=8),
            [EOS_ID],
            clock=FakeClock(),
        )


def test_empty_prompt_is_refused(adapter) -> None:
    with pytest.raises(ConfigError, match="at least one token"):
        decode_target_only(adapter, [], greedy(), [EOS_ID], clock=FakeClock())


def test_eos_terminates_and_nothing_follows_it(adapter) -> None:
    """Invariant I7. The stop token is the last committed token."""
    # Every token is a stop token, so the request must end after exactly one.
    stops = list(range(TINY_VOCAB_SIZE))
    result = decode_target_only(
        adapter,
        PROMPT,
        greedy(eos_policy="respect", max_new_tokens=10),
        stops,
        clock=FakeClock(),
    )
    assert result.termination == "eos"
    assert len(result.output_ids) == 1


def test_suppressed_eos_never_appears_in_the_output(adapter) -> None:
    stops = [EOS_ID, 5, 6, 7]
    result = decode_target_only(
        adapter, PROMPT, greedy(max_new_tokens=20), stops, clock=FakeClock()
    )
    assert result.termination == "budget"
    assert not set(result.output_ids) & set(stops)
    assert len(result.output_ids) == 20


def test_cache_boundary_holds_after_every_committed_token(adapter) -> None:
    sink = ListSink()
    result = decode_target_only(
        adapter, PROMPT, greedy(max_new_tokens=9), [EOS_ID], sink=sink, clock=FakeClock()
    )
    # Invariant I2: the cache ends one token short of the full sequence.
    assert len(sink.events) == 9
    for index, event in enumerate(sink.events):
        assert event.target_cache_after == len(PROMPT) + index
        assert event.draft_cache_after is None
    assert sink.events[-1].target_cache_after == len(PROMPT) + len(result.output_ids) - 1


def test_sequential_requests_do_not_share_state(fixture_model, adapter) -> None:
    """Invariant I10, checked by interleaving two different prompts."""
    other = [30, 31, 32]
    first = decode_target_only(adapter, PROMPT, greedy(), [EOS_ID], clock=FakeClock())
    middle = decode_target_only(adapter, other, greedy(), [EOS_ID], clock=FakeClock())
    again = decode_target_only(adapter, PROMPT, greedy(), [EOS_ID], clock=FakeClock())
    assert first.output_ids == again.output_ids
    assert middle.output_ids != first.output_ids
    assert middle.output_ids == reference_greedy(fixture_model.model, other, 12, {EOS_ID})


# --- sampling --------------------------------------------------------------


def test_sampling_is_reproducible_for_a_given_seed(adapter) -> None:
    config = DecodeConfig("sample", 0.8, 10, "respect", 7)
    left = decode_target_only(adapter, PROMPT, config, [EOS_ID], clock=FakeClock())
    right = decode_target_only(adapter, PROMPT, config, [EOS_ID], clock=FakeClock())
    assert left.output_ids == right.output_ids


def test_different_seeds_give_different_samples(adapter) -> None:
    outputs = {
        seed: decode_target_only(
            adapter,
            PROMPT,
            DecodeConfig("sample", 1.0, 16, "respect", seed),
            [EOS_ID],
            clock=FakeClock(),
        ).output_ids
        for seed in (1, 2, 3, 4)
    }
    assert len(set(outputs.values())) > 1


def test_sampling_ignores_the_checkpoint_generation_config(fixture_model, adapter) -> None:
    """Nothing in this engine reads model.generation_config (CLAUDE.md).

    A top_k of 1 would collapse sampling onto greedy if it were honoured.
    """
    config = DecodeConfig("sample", 1.5, 24, "respect", 5)
    before = decode_target_only(adapter, PROMPT, config, [EOS_ID], clock=FakeClock())
    fixture_model.model.generation_config.top_k = 1
    fixture_model.model.generation_config.temperature = 0.001
    fixture_model.model.generation_config.do_sample = False
    try:
        after = decode_target_only(adapter, PROMPT, config, [EOS_ID], clock=FakeClock())
    finally:
        fixture_model.model.generation_config.top_k = 50
        fixture_model.model.generation_config.temperature = 1.0
        fixture_model.model.generation_config.do_sample = True
    assert after.output_ids == before.output_ids
    greedy_out = decode_target_only(
        adapter,
        PROMPT,
        greedy(max_new_tokens=24, eos_policy="respect"),
        [EOS_ID],
        clock=FakeClock(),
    ).output_ids
    assert after.output_ids != greedy_out


def test_temperature_changes_the_sampled_output(adapter) -> None:
    cold = decode_target_only(
        adapter, PROMPT, DecodeConfig("sample", 0.01, 16, "respect", 3), [EOS_ID], clock=FakeClock()
    )
    hot = decode_target_only(
        adapter, PROMPT, DecodeConfig("sample", 5.0, 16, "respect", 3), [EOS_ID], clock=FakeClock()
    )
    assert cold.output_ids != hot.output_ids
    # A near-zero temperature is effectively greedy.
    assert (
        cold.output_ids
        == decode_target_only(
            adapter,
            PROMPT,
            greedy(max_new_tokens=16, eos_policy="respect"),
            [EOS_ID],
            clock=FakeClock(),
        ).output_ids
    )


def test_sample_mode_without_a_source_is_refused(adapter) -> None:
    row = torch.zeros(TINY_VOCAB_SIZE)
    with pytest.raises(ConfigError, match="random source"):
        select_token(row, DecodeConfig("sample", 1.0, 4, "respect", 1), (), None)


def test_select_token_greedy_consumes_no_randomness() -> None:
    row = torch.tensor([1.0, 5.0, 5.0, 2.0])
    source = TorchRandomSource(request_seed=1)
    assert select_token(row, DecodeConfig("greedy", 1.0, 4, "respect", 1), (), source) == 1


# --- timing boundaries -----------------------------------------------------


def test_timestamps_are_ordered_and_synchronized(adapter) -> None:
    """Invariant I11, with a fake clock so synchronizations can be counted."""
    clock = FakeClock(step_ns=10)
    result = decode_target_only(adapter, PROMPT, greedy(max_new_tokens=5), [EOS_ID], clock=clock)
    assert result.start_ns < result.first_token_ns <= result.last_token_ns <= result.end_ns
    assert list(result.token_release_ns) == sorted(result.token_release_ns)
    # One mark before prefill, one per released token, one at the end.
    assert clock.marks == 1 + len(result.output_ids) + 1


def test_correctness_checks_can_be_disabled_without_changing_output(adapter) -> None:
    config = greedy(max_new_tokens=8)
    checked = decode_target_only(
        adapter,
        PROMPT,
        config,
        [EOS_ID],
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True),
    )
    unchecked = decode_target_only(
        adapter,
        PROMPT,
        config,
        [EOS_ID],
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=False),
    )
    assert checked.output_ids == unchecked.output_ids
    # The setting is recorded, because it must match across compared engines.
    assert checked.counters["correctness_checks"] == 1
    assert unchecked.counters["correctness_checks"] == 0


def test_unavailable_counters_are_null_not_zero(adapter) -> None:
    result = decode_target_only(
        adapter, PROMPT, greedy(max_new_tokens=3), [EOS_ID], clock=FakeClock()
    )
    assert result.counters["draft_cache_length"] is None
    assert result.draft_calls == 0
    assert result.proposed == 0
