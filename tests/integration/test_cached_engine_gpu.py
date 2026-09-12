"""Milestone 3 GPU gate: the native target-only engine on the real Qwen3 pair.

BF16 on a GB10, after the FP32 CPU checks in test_cached_engine_cpu.py.
Marked ``gpu`` and ``download``; a skip never satisfies this gate.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from switchback.decoder import EngineOptions, decode_target_only
from switchback.models.hf_baselines import run_hf_ar
from switchback.models.qwen import QwenAdapter, render_chat_prompt
from switchback.types import DecodeConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.download,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

# A frozen smoke set. These are fixed so the conformance check is the same
# comparison on every run; they are not the locked benchmark cohort.
SMOKE_PROMPTS = (
    "Write a Python function for this task. Return code only. Reverse a string.",
    "Solve the problem and state the final answer. What is 17 times 23?",
    "Write a Python function for this task. Return code only. Sum a list of integers.",
    "Solve the problem and state the final answer. If 3x + 7 = 25, what is x?",
)

# Measured BF16 behaviour on this checkpoint, recorded rather than tuned to pass.
#
# Splitting a prefill into two forward calls changes the reduction order, and
# BF16 keeps 8 mantissa bits. Across 36 layers that produced a maximum absolute
# logit difference of 0.73-1.13 against a max logit magnitude of about 50, i.e.
# roughly 2% relative, with a mean absolute difference near 0.07.
#
# The consequence is worth stating plainly: cached and uncached BF16 logits do
# *not* agree on argmax for every row. One or two rows out of ~30 flip. Every
# observed flip was at a row whose top-2 margin was at most 0.125, so the
# disagreements are genuine near-ties, not misalignment -- a real off-by-one in
# the cache changes logits by order 10 and flips rows with large margins.
# The FP32 CPU test bounds the same comparison at 1e-4.
BF16_RELATIVE_LOGIT_BOUND = 0.05
BF16_TIE_MARGIN_BOUND = 0.5


@pytest.fixture(scope="module")
def target_adapter(loaded_pair: Any) -> QwenAdapter:
    target, _, _ = loaded_pair
    return QwenAdapter(
        model=target.model,
        device="cuda",
        vocab_size=target.logits_vocab_size,
        name="target",
    )


def options() -> EngineOptions:
    return EngineOptions(correctness_checks=True, device="cuda")


@pytest.mark.parametrize("head_width", [1, 14, -3])
def test_cached_logits_match_full_prefix_recomputation(
    loaded_pair: Any, target_adapter: QwenAdapter, head_width: int
) -> None:
    """Cached and uncached BF16 logits, and what their differences may not be.

    A genuine cache misalignment and BF16 reduction-order noise both produce
    "the logits differ". They are told apart by *where* the differences land:
    noise flips only near-ties, misalignment flips confident rows too.
    """
    target, _, _ = loaded_pair
    ids = render_chat_prompt(target.tokenizer, SMOKE_PROMPTS[0])
    split = head_width if head_width > 0 else len(ids) + head_width
    with torch.inference_mode():
        reference = target.model(
            input_ids=torch.tensor([ids], device="cuda"),
            use_cache=False,
        ).logits.float()
        cache = target_adapter.new_cache()
        head = target_adapter.forward(torch.tensor([ids[:split]], device="cuda"), cache)
        tail = target_adapter.forward(torch.tensor([ids[split:]], device="cuda"), cache)
    cached = torch.cat([head, tail], dim=1).float()
    assert cached.shape == reference.shape

    scale = float(reference.abs().max())
    relative = float((cached - reference).abs().max()) / scale
    assert relative < BF16_RELATIVE_LOGIT_BOUND, f"relative difference {relative:.4f}"

    # The decision prefill actually makes must agree.
    assert int(cached[0, -1].argmax()) == int(reference[0, -1].argmax())

    # Any row that does disagree must be a near-tie under the reference.
    disagreeing = (cached.argmax(-1) != reference.argmax(-1))[0]
    top2 = reference[0].topk(2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1])[disagreeing]
    if margins.numel():
        assert float(margins.max()) < BF16_TIE_MARGIN_BOUND, (
            f"{int(disagreeing.sum())} rows disagree, largest top-2 margin "
            f"{float(margins.max()):.3f} on a logit scale of {scale:.1f}"
        )


@pytest.mark.parametrize("prompt", SMOKE_PROMPTS)
def test_native_greedy_matches_the_hf_baseline_token_for_token(
    loaded_pair: Any, target_adapter: QwenAdapter, prompt: str
) -> None:
    """Invariant I6 on the real model: token-ID arrays, not decoded text."""
    target, _, _ = loaded_pair
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=32,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    ids = render_chat_prompt(target.tokenizer, prompt)
    baseline = run_hf_ar(
        target.model, ids, config, target.eos_token_ids, target.tokenizer.pad_token_id, "cuda"
    )
    native = decode_target_only(
        target_adapter, ids, config, target.eos_token_ids, options=options()
    )
    assert native.output_ids == baseline.output_ids
    assert native.target_calls == 32


def test_native_respects_eos_like_the_baseline(
    loaded_pair: Any, target_adapter: QwenAdapter
) -> None:
    target, _, _ = loaded_pair
    config = DecodeConfig(
        mode="greedy", temperature=1.0, max_new_tokens=64, eos_policy="respect", seed=42
    )
    ids = render_chat_prompt(target.tokenizer, "Reply with exactly: ok")
    baseline = run_hf_ar(
        target.model, ids, config, target.eos_token_ids, target.tokenizer.pad_token_id, "cuda"
    )
    native = decode_target_only(
        target_adapter, ids, config, target.eos_token_ids, options=options()
    )
    assert native.output_ids == baseline.output_ids
    if native.termination == "eos":
        assert native.output_ids[-1] in target.eos_token_ids
        assert not set(native.output_ids[:-1]) & set(target.eos_token_ids)


def test_a_rollback_leaves_the_cache_byte_identical_in_effect(
    loaded_pair: Any, target_adapter: QwenAdapter
) -> None:
    target, _, _ = loaded_pair
    ids = render_chat_prompt(target.tokenizer, SMOKE_PROMPTS[1])
    with torch.inference_mode():
        cache = target_adapter.new_cache()
        target_adapter.forward(torch.tensor([ids], device="cuda"), cache)
        expected = target_adapter.forward(torch.tensor([[3838, 374]], device="cuda"), cache)
        target_adapter.crop(cache, len(ids))
        target_adapter.forward(torch.tensor([[100, 200, 300]], device="cuda"), cache)
        target_adapter.crop(cache, len(ids))
        replayed = target_adapter.forward(torch.tensor([[3838, 374]], device="cuda"), cache)
    assert torch.equal(replayed, expected)
    assert cache.length == len(ids) + 2


def test_requests_are_independent_in_sequence(
    loaded_pair: Any, target_adapter: QwenAdapter
) -> None:
    target, _, _ = loaded_pair
    config = DecodeConfig("greedy", 1.0, 16, "suppress_until_budget", 42)
    first_ids = render_chat_prompt(target.tokenizer, SMOKE_PROMPTS[0])
    other_ids = render_chat_prompt(target.tokenizer, SMOKE_PROMPTS[2])
    first = decode_target_only(
        target_adapter, first_ids, config, target.eos_token_ids, options=options()
    )
    decode_target_only(target_adapter, other_ids, config, target.eos_token_ids, options=options())
    again = decode_target_only(
        target_adapter, first_ids, config, target.eos_token_ids, options=options()
    )
    assert first.output_ids == again.output_ids


def test_timings_are_synchronized_and_ordered(
    loaded_pair: Any, target_adapter: QwenAdapter
) -> None:
    target, _, _ = loaded_pair
    config = DecodeConfig("greedy", 1.0, 8, "suppress_until_budget", 42)
    ids = render_chat_prompt(target.tokenizer, SMOKE_PROMPTS[3])
    result = decode_target_only(
        target_adapter, ids, config, target.eos_token_ids, options=options()
    )
    assert result.start_ns < result.first_token_ns <= result.last_token_ns <= result.end_ns
    assert list(result.token_release_ns) == sorted(result.token_release_ns)
    # A real decoding step cannot take zero nanoseconds on this hardware; a zero
    # would mean the clock never synchronized.
    assert result.last_token_ns > result.first_token_ns
