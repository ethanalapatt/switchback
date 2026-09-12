"""Milestone 5 GPU gate: sampled speculation on the real Qwen3 pair.

Sampled output is *not* required to match any baseline token for token, even at
the same seed (SPEC.md section 4.1). What is required: the same engine, config
and seed repeats; a rejection actually rolls the cache back correctly; and the
engine samples from the specified distribution rather than a truncated one.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from switchback.decoder import (
    EngineOptions,
    decode_speculative_sampled,
    decode_target_only,
)
from switchback.events import ListSink
from switchback.models.qwen import QwenAdapter, render_chat_prompt
from switchback.traces import block_summary, build_header, replay_trace, write_trace
from switchback.types import DecodeConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.download,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

PROMPT = "Write a Python function for this task. Return code only. Reverse a string."


@pytest.fixture(scope="module")
def pair_adapters(loaded_pair: Any):
    target, draft, _ = loaded_pair
    return (
        QwenAdapter(
            model=target.model,
            device="cuda",
            vocab_size=target.logits_vocab_size,
            name="target",
        ),
        QwenAdapter(
            model=draft.model,
            device="cuda",
            vocab_size=draft.logits_vocab_size,
            name="draft",
        ),
    )


def options() -> EngineOptions:
    return EngineOptions(correctness_checks=True, device="cuda")


def config(**overrides) -> DecodeConfig:
    base = dict(
        mode="sample",
        temperature=0.7,
        max_new_tokens=48,
        eos_policy="suppress_until_budget",
        seed=11,
    )
    base.update(overrides)
    return DecodeConfig(**base)


def test_the_same_seed_repeats(loaded_pair: Any, pair_adapters) -> None:
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, PROMPT)
    outputs = {
        decode_speculative_sampled(
            target,
            draft,
            ids,
            config(),
            target_model.eos_token_ids,
            gamma=4,
            options=options(),
        ).output_ids
        for _ in range(3)
    }
    assert len(outputs) == 1


def test_different_seeds_diverge(loaded_pair: Any, pair_adapters) -> None:
    """Seeds must be able to change the output, but need not always.

    Requiring three seeds to give three distinct completions failed here, and
    correctly so: at temperature 0.7 on "reverse a string" the target
    distribution is peaked enough that two seeds produced the same 48 tokens.
    That is the model being confident, not the RNG being broken. The property
    worth asserting is that the seed reaches the sampler at all, so this uses a
    temperature with real entropy and asks only for more than one completion.
    """
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, PROMPT)
    outputs = {
        decode_speculative_sampled(
            target,
            draft,
            ids,
            config(seed=seed, temperature=1.0, max_new_tokens=32),
            target_model.eos_token_ids,
            gamma=4,
            options=options(),
        ).output_ids
        for seed in (11, 22, 33, 44)
    }
    assert len(outputs) > 1


def test_sampling_is_not_secretly_greedy(loaded_pair: Any, pair_adapters) -> None:
    """At temperature 1.0 the output must differ from argmax decoding.

    If a truncation default had survived, or if the residual draw had been
    replaced by an argmax, this is where it would show.
    """
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, PROMPT)
    greedy = decode_target_only(
        target,
        ids,
        DecodeConfig("greedy", 1.0, 48, "suppress_until_budget", 11),
        target_model.eos_token_ids,
        options=options(),
    )
    sampled = decode_speculative_sampled(
        target,
        draft,
        ids,
        config(temperature=1.0),
        target_model.eos_token_ids,
        gamma=4,
        options=options(),
    )
    assert sampled.output_ids != greedy.output_ids
    assert len(sampled.output_ids) == 48


def test_a_low_temperature_approaches_greedy(loaded_pair: Any, pair_adapters) -> None:
    """The converse check: the temperature is genuinely being applied."""
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, PROMPT)
    greedy = decode_target_only(
        target,
        ids,
        DecodeConfig("greedy", 1.0, 32, "suppress_until_budget", 11),
        target_model.eos_token_ids,
        options=options(),
    )
    cold = decode_speculative_sampled(
        target,
        draft,
        ids,
        config(temperature=0.01, max_new_tokens=32),
        target_model.eos_token_ids,
        gamma=4,
        options=options(),
    )
    matching = sum(
        1 for left, right in zip(cold.output_ids, greedy.output_ids, strict=True) if left == right
    )
    assert matching >= 30, f"{matching}/32 matched greedy at temperature 0.01"


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_cache_boundaries_hold_through_sampled_rejections(
    loaded_pair: Any, pair_adapters, gamma: int
) -> None:
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, PROMPT)
    sink = ListSink()
    result = decode_speculative_sampled(
        target,
        draft,
        ids,
        config(max_new_tokens=64),
        target_model.eos_token_ids,
        gamma=gamma,
        sink=sink,
        options=options(),
    )
    committed = 0
    for event in sink.events:
        committed += event.committed
        assert event.target_cache_after == len(ids) + committed - 1
        if event.action == "speculative":
            assert event.draft_cache_after == len(ids) + committed - 1
    assert committed == len(result.output_ids)
    summary = block_summary([event.as_dict() for event in sink.events])
    # Sampled acceptance is lower than greedy; a run with no rejection at all
    # would leave the sampled rollback path untested here.
    assert summary["rejected_blocks"] > 0


def test_a_saved_sampled_trace_replays(loaded_pair: Any, pair_adapters, tmp_path) -> None:
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, PROMPT)
    sink = ListSink()
    decode_config = config(max_new_tokens=48, eos_policy="respect")
    result = decode_speculative_sampled(
        target,
        draft,
        ids,
        decode_config,
        target_model.eos_token_ids,
        gamma=4,
        sink=sink,
        options=options(),
    )
    header = build_header(
        engine="fixed_4",
        result=result,
        prompt_ids=ids,
        mode="sample",
        eos_policy="respect",
        temperature=decode_config.temperature,
        seed=decode_config.seed,
        gamma=4,
        max_new_tokens=48,
    )
    path = tmp_path / "sampled.jsonl"
    write_trace(path, header, sink)

    from switchback.traces import read_trace

    loaded_header, blocks = read_trace(path)
    report = replay_trace(loaded_header, blocks)
    assert report.ok, report.problems
    assert report.committed_tokens == len(result.output_ids)


def test_acceptance_is_lower_than_greedy_but_not_zero(loaded_pair: Any, pair_adapters) -> None:
    """Recorded, not asserted as a performance result."""
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, PROMPT)
    result = decode_speculative_sampled(
        target,
        draft,
        ids,
        config(temperature=0.7, max_new_tokens=64),
        target_model.eos_token_ids,
        gamma=4,
        options=options(),
    )
    assert result.proposed > 0
    assert 0 < result.accepted < result.proposed


def test_eos_is_respected_and_suppressed_as_configured(loaded_pair: Any, pair_adapters) -> None:
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, "Reply with exactly: ok")
    respected = decode_speculative_sampled(
        target,
        draft,
        ids,
        config(eos_policy="respect", max_new_tokens=64, temperature=0.1),
        target_model.eos_token_ids,
        gamma=4,
        options=options(),
    )
    if respected.termination == "eos":
        assert respected.output_ids[-1] in target_model.eos_token_ids
        assert not set(respected.output_ids[:-1]) & set(target_model.eos_token_ids)
    suppressed = decode_speculative_sampled(
        target,
        draft,
        ids,
        config(eos_policy="suppress_until_budget", max_new_tokens=32),
        target_model.eos_token_ids,
        gamma=4,
        options=options(),
    )
    assert len(suppressed.output_ids) == 32
    assert not set(suppressed.output_ids) & set(target_model.eos_token_ids)
