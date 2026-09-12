"""Milestone 1 pilot: measure the two Hugging Face baselines on real hardware.

The pilot exists to answer "does the model pair run here, and roughly how long
does a request take", so milestone 7's full matrix can be scheduled. It writes
raw per-request timings and computes **no** speedup: a handful of unrandomized
requests is not a benchmark cohort (SPEC.md section 9.3).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchback.decoder import EngineOptions, decode_target_only
from switchback.models import hf_baselines
from switchback.models.qwen import (
    LoadedModel,
    QwenAdapter,
    check_logits_vocab_match,
    check_tokenizer_compatibility,
    load_qwen,
    render_chat_prompt,
)
from switchback.provenance import collect_source_provenance
from switchback.types import DecodeConfig, TokenizerCompatibility

# Short, self-contained instructions. They are pilot smoke prompts, not the
# locked MBPP/GSM8K cohort, and are never reported as a benchmark result.
PILOT_PROMPTS: tuple[str, ...] = (
    "Write a Python function for this task. Return code only. "
    "Return the sum of the even numbers in a list.",
    "Solve the problem and state the final answer. "
    "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
    "Write a Python function for this task. Return code only. "
    "Check whether a string is a palindrome, ignoring case and spaces.",
    "Solve the problem and state the final answer. If 3x + 7 = 25, what is x?",
)


@dataclass(frozen=True)
class PilotRequest:
    """One measured pilot request. Field names carry their units."""

    engine: str
    prompt_index: int
    repeat: int
    prompt_tokens: int
    output_tokens: int
    latency_ns: int
    ttft_ns: int
    decode_ns: int
    termination: str
    output_ids: list[int]
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None


def load_pair(
    target_repo: str,
    target_revision: str,
    draft_repo: str,
    draft_revision: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    local_files_only: bool = False,
) -> tuple[LoadedModel, LoadedModel, TokenizerCompatibility, dict[str, float]]:
    """Load target and draft, then verify they can share one token stream."""
    timings: dict[str, float] = {}
    started = time.perf_counter()
    target = load_qwen(
        target_repo,
        target_revision,
        dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    timings["target_load_s"] = time.perf_counter() - started
    started = time.perf_counter()
    draft = load_qwen(
        draft_repo,
        draft_revision,
        dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    timings["draft_load_s"] = time.perf_counter() - started
    compatibility = check_tokenizer_compatibility(
        target.tokenizer,
        draft.tokenizer,
        int(target.config.vocab_size),
        int(draft.config.vocab_size),
    )
    compatibility.raise_if_incompatible()
    check_logits_vocab_match(target, draft)
    return target, draft, compatibility, timings


def _memory_counters(device: str) -> tuple[int | None, int | None]:
    import torch

    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _reset_memory_counters(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def run_pilot(
    target: LoadedModel,
    draft: LoadedModel,
    device: str,
    max_new_tokens: int = 32,
    repeats: int = 3,
    warmups: int = 2,
    prompts: tuple[str, ...] = PILOT_PROMPTS,
    seed: int = 42,
) -> tuple[list[PilotRequest], list[str]]:
    """Run ``hf_ar``, ``native_ar`` and ``hf_dynamic`` over the pilot prompts.

    ``native_ar`` is included so the pilot can detect the risk SPEC.md section 11
    names first: Python-level overhead in Switchback's own loop dominating any
    saving from drafting. Returns the measured requests plus any greedy output
    mismatch against ``hf_ar``.
    """
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=max_new_tokens,
        eos_policy="suppress_until_budget",
        seed=seed,
    )
    eos_ids = target.eos_token_ids
    pad_id = getattr(target.tokenizer, "pad_token_id", None)
    prompt_ids = [render_chat_prompt(target.tokenizer, text) for text in prompts]

    adapter = QwenAdapter(
        model=target.model,
        device=device,
        vocab_size=target.logits_vocab_size,
        name="target",
    )
    options = EngineOptions(correctness_checks=True, device=device)

    def run_native(ids: list[int]) -> hf_baselines.BaselineResult:
        """Adapt a DecodeResult to the baseline record so the columns line up."""
        result = decode_target_only(adapter, ids, config, eos_ids, options=options)
        return hf_baselines.BaselineResult(
            engine="native_ar",
            output_ids=result.output_ids,
            token_release_ns=result.token_release_ns,
            start_ns=result.start_ns,
            first_token_ns=result.first_token_ns,
            last_token_ns=result.last_token_ns,
            end_ns=result.end_ns,
            prompt_tokens=len(ids),
            termination=result.termination,
            generation_config={"engine": "switchback native target-only"},
            counters={
                "accepted": result.accepted,
                "proposed": result.proposed,
                "target_calls": result.target_calls,
                "bypass_decisions": result.bypass_decisions,
                "controller_decisions": result.controller_decisions,
            },
        )

    engines = {
        "hf_ar": lambda ids: hf_baselines.run_hf_ar(
            target.model, ids, config, eos_ids, pad_id, device
        ),
        "native_ar": run_native,
        "hf_dynamic": lambda ids: hf_baselines.run_hf_dynamic(
            target.model, draft.model, ids, config, eos_ids, pad_id, device
        ),
    }

    # Warm kernels and allocator state before anything is recorded. Warmup cost
    # is excluded from warm-request latency by SPEC.md section 9.4.
    for _ in range(warmups):
        for run in engines.values():
            run(prompt_ids[0])

    measured: list[PilotRequest] = []
    outputs: dict[tuple[int, str], tuple[int, ...]] = {}
    for repeat in range(repeats):
        for index, ids in enumerate(prompt_ids):
            for engine, run in engines.items():
                _reset_memory_counters(device)
                result = run(ids)
                allocated, reserved = _memory_counters(device)
                outputs.setdefault((index, engine), result.output_ids)
                decode_ns = result.last_token_ns - result.first_token_ns
                measured.append(
                    PilotRequest(
                        engine=engine,
                        prompt_index=index,
                        repeat=repeat,
                        prompt_tokens=result.prompt_tokens,
                        output_tokens=len(result.output_ids),
                        latency_ns=result.end_ns - result.start_ns,
                        ttft_ns=result.first_token_ns - result.start_ns,
                        decode_ns=decode_ns,
                        termination=result.termination,
                        output_ids=list(result.output_ids),
                        peak_allocated_bytes=allocated,
                        peak_reserved_bytes=reserved,
                    )
                )
    # Greedy equality is required across all three engines before any timing
    # comparison means anything (invariant I6). Mismatches are recorded, never
    # smoothed over.
    mismatches = [
        f"prompt {index}: {engine} output differs from hf_ar"
        for index in range(len(prompt_ids))
        for engine in ("native_ar", "hf_dynamic")
        if outputs.get((index, "hf_ar")) != outputs.get((index, engine))
    ]
    return measured, mismatches


def write_pilot(
    out: Path,
    requests: list[PilotRequest],
    mismatches: list[str],
    target: LoadedModel,
    draft: LoadedModel,
    compatibility: TokenizerCompatibility,
    load_timings: dict[str, float],
    device: str,
    max_new_tokens: int,
    repeats: int,
    environment: dict[str, Any],
) -> None:
    """Persist the pilot as raw evidence, atomically."""
    provenance = collect_source_provenance()
    document = {
        "kind": "pilot",
        "data_kind": "measured",
        "is_benchmark_result": False,
        "disclaimer": (
            "Pilot timings size the milestone 7 benchmark. The prompt set is not "
            "the locked cohort, engine order is not randomized, and no speedup is "
            "derived from these numbers."
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": device,
        "condition": {
            "mode": "greedy",
            "eos_policy": "suppress_until_budget",
            "max_new_tokens": max_new_tokens,
            "repeats": repeats,
            "seed": 42,
        },
        "models": {
            "target": {**asdict(target.spec), "parameters": target.parameter_count},
            "draft": {**asdict(draft.spec), "parameters": draft.parameter_count},
        },
        "tokenizer_compatibility": asdict(compatibility),
        "setup_seconds": load_timings,
        "source": {
            "commit": provenance.commit,
            "dirty": provenance.dirty,
            "source_sha256": provenance.source_sha256,
        },
        "environment": environment,
        "greedy_output_mismatches": mismatches,
        "requests": [asdict(request) for request in requests],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, out)


def summarize(requests: list[PilotRequest]) -> str:
    """Per-engine medians for the terminal. Descriptive only, never a claim."""
    import statistics

    lines = [
        "  engine       n   median_latency_ms  median_ttft_ms  output_tokens",
        "  " + "-" * 62,
    ]
    for engine in sorted({request.engine for request in requests}):
        sample = [request for request in requests if request.engine == engine]
        latency_ms = statistics.median(request.latency_ns / 1e6 for request in sample)
        ttft_ms = statistics.median(request.ttft_ns / 1e6 for request in sample)
        tokens = sorted({request.output_tokens for request in sample})
        lines.append(
            f"  {engine:<11} {len(sample):>3}   {latency_ms:>16.1f}  {ttft_ms:>14.1f}  {tokens}"
        )
    lines.append("")
    lines.append("  Pilot only: not a benchmark cohort and not a speedup measurement.")
    return "\n".join(lines)
