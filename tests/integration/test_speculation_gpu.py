"""Milestone 4 GPU gate: fixed greedy speculation on the real Qwen3 pair.

The acceptance criterion is exact token-ID agreement with target-only decoding
at every fixed length, at two context lengths. Marked ``gpu``; a skip is a skip.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from switchback.decoder import (
    EngineOptions,
    decode_speculative_greedy,
    decode_target_only,
)
from switchback.events import ListSink
from switchback.models.hf_baselines import run_hf_ar
from switchback.models.qwen import QwenAdapter, render_chat_prompt
from switchback.types import DecodeConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.download,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

GAMMAS = (1, 2, 4, 8)

SHORT_PROMPT = "Write a Python function for this task. Return code only. Reverse a string."
LONG_PROMPT = (
    "Write a Python function for this task. Return code only. "
    "Given a list of dictionaries describing employees, each with the keys name, "
    "department, salary and start_date, group the employees by department, compute "
    "the mean salary per department, sort the departments by that mean in descending "
    "order, and return a list of tuples of department name and rounded mean salary. "
    "Handle an empty input list by returning an empty list, and ignore any employee "
    "record that is missing the salary key rather than raising."
)


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
        mode="greedy",
        temperature=1.0,
        max_new_tokens=32,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    base.update(overrides)
    return DecodeConfig(**base)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("prompt", [SHORT_PROMPT, LONG_PROMPT], ids=["short", "long"])
def test_fixed_speculation_is_token_identical_to_target_only(
    loaded_pair: Any, pair_adapters, gamma: int, prompt: str
) -> None:
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, prompt)
    decode_config = config()
    reference = decode_target_only(
        target, ids, decode_config, target_model.eos_token_ids, options=options()
    )
    result = decode_speculative_greedy(
        target,
        draft,
        ids,
        decode_config,
        target_model.eos_token_ids,
        gamma=gamma,
        options=options(),
    )
    assert result.output_ids == reference.output_ids


@pytest.mark.parametrize("gamma", GAMMAS)
def test_fixed_speculation_also_matches_the_hf_baseline(
    loaded_pair: Any, pair_adapters, gamma: int
) -> None:
    """Invariant I6 against the external reference, not just our own engine."""
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, SHORT_PROMPT)
    decode_config = config()
    baseline = run_hf_ar(
        target_model.model,
        ids,
        decode_config,
        target_model.eos_token_ids,
        target_model.tokenizer.pad_token_id,
        "cuda",
    )
    result = decode_speculative_greedy(
        target,
        draft,
        ids,
        decode_config,
        target_model.eos_token_ids,
        gamma=gamma,
        options=options(),
    )
    assert result.output_ids == baseline.output_ids


@pytest.mark.parametrize("gamma", GAMMAS)
def test_acceptance_is_nonzero_on_a_same_family_draft(
    loaded_pair: Any, pair_adapters, gamma: int
) -> None:
    """Qwen3-0.6B should agree with Qwen3-4B often enough to be worth drafting.

    This records the acceptance rate; it does not assert a performance result.
    """
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, SHORT_PROMPT)
    result = decode_speculative_greedy(
        target,
        draft,
        ids,
        config(max_new_tokens=64),
        target_model.eos_token_ids,
        gamma=gamma,
        options=options(),
    )
    assert result.proposed > 0
    assert result.accepted > 0
    # Fewer target calls than committed tokens is the mechanism by which
    # speculation can save time. Whether it does is a milestone 7 question.
    assert result.target_calls < len(result.output_ids)


@pytest.mark.parametrize("gamma", [1, 4, 8])
def test_cache_boundaries_hold_throughout_a_real_request(
    loaded_pair: Any, pair_adapters, gamma: int
) -> None:
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, SHORT_PROMPT)
    sink = ListSink()
    result = decode_speculative_greedy(
        target,
        draft,
        ids,
        config(max_new_tokens=48),
        target_model.eos_token_ids,
        gamma=gamma,
        sink=sink,
        options=options(),
    )
    committed = 0
    saw_rejection = False
    for event in sink.events:
        committed += event.committed
        assert event.target_cache_after == len(ids) + committed - 1
        if event.action == "speculative":
            assert event.draft_cache_after == len(ids) + committed - 1
            saw_rejection = saw_rejection or event.rejection_position is not None
    assert committed == len(result.output_ids)
    # A real draft is wrong sometimes; if it never were, the rollback path would
    # be untested here and this assertion would say so.
    assert saw_rejection


def test_eos_termination_matches_target_only(loaded_pair: Any, pair_adapters) -> None:
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, "Reply with exactly: ok")
    decode_config = config(eos_policy="respect", max_new_tokens=64)
    reference = decode_target_only(
        target, ids, decode_config, target_model.eos_token_ids, options=options()
    )
    for gamma in GAMMAS:
        result = decode_speculative_greedy(
            target,
            draft,
            ids,
            decode_config,
            target_model.eos_token_ids,
            gamma=gamma,
            options=options(),
        )
        assert result.output_ids == reference.output_ids
        assert result.termination == reference.termination


def test_timings_bound_all_of_the_request(loaded_pair: Any, pair_adapters) -> None:
    target_model, _, _ = loaded_pair
    target, draft = pair_adapters
    ids = render_chat_prompt(target_model.tokenizer, SHORT_PROMPT)
    result = decode_speculative_greedy(
        target,
        draft,
        ids,
        config(max_new_tokens=16),
        target_model.eos_token_ids,
        gamma=4,
        options=options(),
    )
    assert result.start_ns < result.first_token_ns <= result.last_token_ns <= result.end_ns
    assert list(result.token_release_ns) == sorted(result.token_release_ns)
    # A verified block releases several tokens at once, so repeated timestamps
    # are expected and are recorded as measured rather than spread out.
    assert len(result.token_release_ns) == len(result.output_ids)
    assert len(set(result.token_release_ns)) < len(result.token_release_ns)
