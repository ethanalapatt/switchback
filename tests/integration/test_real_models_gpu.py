"""Milestone 1 GPU gate: the pinned Qwen3 pair actually runs on this machine.

Marked ``gpu`` and ``download``. A skip is a skip: it never satisfies the gate.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from switchback.models.hf_baselines import run_hf_ar, run_hf_dynamic
from switchback.models.qwen import (
    assert_resident,
    check_logits_vocab_match,
    render_chat_prompt,
)
from switchback.types import MAX_CONTEXT_TOKENS, DecodeConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.download,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

SMOKE_PROMPT = "Write a Python function for this task. Return code only. Reverse a string."


def test_both_models_are_resident_with_no_offload(loaded_pair: Any) -> None:
    target, draft, _ = loaded_pair
    assert_resident(target.model, "cuda")
    assert_resident(draft.model, "cuda")
    assert getattr(target.model, "hf_device_map", None) in (None, {})
    assert target.parameter_count > draft.parameter_count


def test_weights_are_bfloat16(loaded_pair: Any) -> None:
    target, draft, _ = loaded_pair
    for model in (target.model, draft.model):
        dtypes = {parameter.dtype for parameter in model.parameters()}
        assert dtypes == {torch.bfloat16}, dtypes


def test_tokenizer_parity_holds_for_the_real_pair(loaded_pair: Any) -> None:
    target, draft, compatibility = loaded_pair
    assert compatibility.compatible, compatibility.failures
    assert compatibility.target_vocab_size == compatibility.draft_vocab_size
    assert compatibility.checked_probe_strings >= 20
    check_logits_vocab_match(target, draft)


def test_both_checkpoints_cover_the_context_bound(loaded_pair: Any) -> None:
    target, draft, _ = loaded_pair
    assert target.max_position_embeddings >= MAX_CONTEXT_TOKENS
    assert draft.max_position_embeddings >= MAX_CONTEXT_TOKENS


def test_chat_template_is_applied_without_thinking_mode(loaded_pair: Any) -> None:
    target, _, _ = loaded_pair
    ids = render_chat_prompt(target.tokenizer, SMOKE_PROMPT)
    text = target.tokenizer.decode(ids)
    assert ids and all(isinstance(value, int) for value in ids)
    assert "<|im_start|>user" in text
    assert text.rstrip().endswith("<|im_start|>assistant") or "assistant" in text
    assert "<think>" not in text or "</think>" in text


@pytest.mark.parametrize("engine", ["hf_ar", "hf_dynamic"])
def test_thirty_two_token_smoke_request_finishes(loaded_pair: Any, engine: str) -> None:
    target, draft, _ = loaded_pair
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=32,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    ids = render_chat_prompt(target.tokenizer, SMOKE_PROMPT)
    pad = target.tokenizer.pad_token_id
    if engine == "hf_ar":
        result = run_hf_ar(target.model, ids, config, target.eos_token_ids, pad, "cuda")
    else:
        result = run_hf_dynamic(
            target.model, draft.model, ids, config, target.eos_token_ids, pad, "cuda"
        )
    assert len(result.output_ids) == 32
    assert result.start_ns < result.first_token_ns <= result.last_token_ns <= result.end_ns
    assert len(result.token_release_ns) == 32


def test_suppressed_eos_produces_the_exact_budget_for_both_engines(
    loaded_pair: Any,
) -> None:
    # The fixed-length condition is only fair if every engine emits the same
    # number of tokens; Qwen3's own generation_config would otherwise stop early.
    target, draft, _ = loaded_pair
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=24,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    ids = render_chat_prompt(target.tokenizer, "Say hi.")
    pad = target.tokenizer.pad_token_id
    left = run_hf_ar(target.model, ids, config, target.eos_token_ids, pad, "cuda")
    right = run_hf_dynamic(
        target.model, draft.model, ids, config, target.eos_token_ids, pad, "cuda"
    )
    assert len(left.output_ids) == len(right.output_ids) == 24


def test_assisted_generation_does_not_change_greedy_tokens(loaded_pair: Any) -> None:
    """Speculation is exact under greedy decoding, so the IDs must match.

    This compares token-ID arrays, not decoded text (invariant I6).
    """
    target, draft, _ = loaded_pair
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=32,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    ids = render_chat_prompt(target.tokenizer, SMOKE_PROMPT)
    pad = target.tokenizer.pad_token_id
    baseline = run_hf_ar(target.model, ids, config, target.eos_token_ids, pad, "cuda")
    assisted = run_hf_dynamic(
        target.model, draft.model, ids, config, target.eos_token_ids, pad, "cuda"
    )
    assert assisted.output_ids == baseline.output_ids


def test_the_assistant_generation_config_is_restored_after_a_request(
    loaded_pair: Any,
) -> None:
    # Assisted generation mutates the assistant's config; leaking that state
    # would make request N+1 depend on request N (invariant I10).
    target, draft, _ = loaded_pair
    before = draft.model.generation_config.to_dict()
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=8,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    ids = render_chat_prompt(target.tokenizer, "Say hi.")
    run_hf_dynamic(
        target.model,
        draft.model,
        ids,
        config,
        target.eos_token_ids,
        target.tokenizer.pad_token_id,
        "cuda",
    )
    assert draft.model.generation_config.to_dict() == before


def test_eos_is_respected_when_the_policy_asks_for_it(loaded_pair: Any) -> None:
    target, _, _ = loaded_pair
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=256,
        eos_policy="respect",
        seed=42,
    )
    ids = render_chat_prompt(target.tokenizer, "Reply with exactly: ok")
    result = run_hf_ar(
        target.model, ids, config, target.eos_token_ids, target.tokenizer.pad_token_id, "cuda"
    )
    assert len(result.output_ids) <= 256
    if result.termination == "eos":
        assert result.output_ids[-1] in target.eos_token_ids
        assert not any(value in target.eos_token_ids for value in result.output_ids[:-1])
