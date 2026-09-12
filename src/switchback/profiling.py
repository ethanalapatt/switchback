"""Per-stage diagnostic timing for one speculative block.

This is a **separate diagnostic pass**, never the measured path (SPEC.md section
9.4). It synchronizes between stages so each one can be attributed, and those
extra synchronizations cost real time: the sum of the stages here is larger than
an uninstrumented block, and the difference is reported rather than hidden.

Its purpose is the question SPEC.md section 11 puts first -- is Python-level
overhead, rather than model compute, deciding the result -- and the cost model
milestone 6 needs.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchback.decoder import _verify_greedy
from switchback.models.qwen import QwenAdapter
from switchback.provenance import collect_source_provenance
from switchback.sampling import forbidden_token_ids, greedy_token
from switchback.types import DecodeConfig


@dataclass
class StageSamples:
    """Per-block durations for one stage, in nanoseconds."""

    name: str
    samples: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {"count": 0, "median_ns": None, "mean_ns": None, "min_ns": None, "max_ns": None}
        return {
            "count": len(self.samples),
            "median_ns": statistics.median(self.samples),
            "mean_ns": statistics.fmean(self.samples),
            "min_ns": min(self.samples),
            "max_ns": max(self.samples),
        }


def _sync(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device if device != "cuda" else None)


def profile_speculative_block(
    target: QwenAdapter,
    draft: QwenAdapter,
    prompt_ids: Sequence[int],
    config: DecodeConfig,
    eos_token_ids: Sequence[int],
    gamma: int,
    blocks: int = 16,
    warmup_blocks: int = 2,
    device: str = "cuda",
) -> dict[str, Any]:
    """Attribute one block's time to drafting, verification, and bookkeeping.

    Reimplements the block here rather than instrumenting ``decoder.py``, so the
    production loop carries no profiling branches at all.
    """
    import torch

    forbidden = forbidden_token_ids(config, eos_token_ids)
    stages = {
        name: StageSamples(name)
        for name in (
            "draft_proposals",
            "target_verification",
            "commit_and_crop",
            "draft_catchup",
            "whole_block",
        )
    }
    accepted_total = 0
    proposed_total = 0
    catchups = 0
    prompt = [int(value) for value in prompt_ids]

    target.checks = False
    draft.checks = False
    target_cache = target.new_cache()
    draft_cache = draft.new_cache()

    with torch.inference_mode():
        logits = target.forward(
            torch.tensor([prompt], dtype=torch.long, device=device), target_cache
        )
        sequence = [*prompt, greedy_token(logits[0, -1], forbidden_ids=forbidden)]
        draft.forward(torch.tensor([prompt], dtype=torch.long, device=device), draft_cache)

        for index in range(blocks + warmup_blocks):
            measuring = index >= warmup_blocks
            _sync(device)
            block_start = time.perf_counter_ns()

            proposals: list[int] = []
            pending = sequence[-1]
            for _ in range(gamma):
                row = draft.forward(
                    torch.tensor([[pending]], dtype=torch.long, device=device), draft_cache
                )
                pending = greedy_token(row[0, -1], forbidden_ids=forbidden)
                proposals.append(pending)
            _sync(device)
            after_draft = time.perf_counter_ns()

            rows = target.forward(
                torch.tensor([[sequence[-1], *proposals]], dtype=torch.long, device=device),
                target_cache,
            )
            _sync(device)
            after_target = time.perf_counter_ns()

            outcome = _verify_greedy(rows, proposals, forbidden)
            keep = len(sequence) + outcome.accepted
            target.crop(target_cache, keep)
            if not outcome.all_accepted:
                draft.crop(draft_cache, keep)
            sequence.extend(outcome.emitted)
            _sync(device)
            after_commit = time.perf_counter_ns()

            # The catch-up forward is a separate stage. Folding it into
            # "commit and crop" made bookkeeping look like it cost a whole draft
            # call, which is how a 0.5 ms stage came to read as 12 ms.
            if outcome.all_accepted:
                draft.forward(
                    torch.tensor([[proposals[-1]]], dtype=torch.long, device=device), draft_cache
                )
                _sync(device)
                if measuring:
                    catchups += 1
            after_catchup = time.perf_counter_ns()

            if measuring:
                accepted_total += outcome.accepted
                proposed_total += len(proposals)
                stages["draft_proposals"].samples.append(after_draft - block_start)
                stages["target_verification"].samples.append(after_target - after_draft)
                stages["commit_and_crop"].samples.append(after_commit - after_target)
                stages["draft_catchup"].samples.append(after_catchup - after_commit)
                stages["whole_block"].samples.append(after_catchup - block_start)

    return {
        "gamma": gamma,
        "blocks_measured": blocks,
        "warmup_blocks": warmup_blocks,
        "prompt_tokens": len(prompt),
        "stages": {name: stage.summary() for name, stage in stages.items()},
        "committed_tokens": len(sequence) - len(prompt),
        "proposed": proposed_total,
        "accepted": accepted_total,
        "acceptance_rate": (accepted_total / proposed_total) if proposed_total else None,
        "catchup_blocks": catchups,
    }


def profile_single_forward(
    adapter: QwenAdapter,
    prompt_ids: Sequence[int],
    widths: Sequence[int],
    repeats: int = 24,
    warmups: int = 4,
    device: str = "cuda",
) -> dict[str, Any]:
    """Cost of one forward call at several widths, from a warm cache.

    The verification width is the lever speculation pulls: if a width of nine
    costs the same as a width of one, drafting is nearly free on the target side
    and the draft's own cost decides the outcome.
    """
    import torch

    adapter.checks = False
    prompt = [int(value) for value in prompt_ids]
    results: dict[str, Any] = {}

    # One global warmup before the width loop. Without it the first width
    # measured also pays the one-time allocator and kernel setup, which made
    # width 1 read slower than width 2 and would bias the controller toward
    # drafting by inflating the target-only cost it compares against.
    with torch.inference_mode():
        warm = adapter.new_cache()
        adapter.forward(torch.tensor([prompt], dtype=torch.long, device=device), warm)
        base_warm = adapter.cache_length(warm)
        for _ in range(warmups):
            adapter.forward(
                torch.tensor([[1] * max(widths)], dtype=torch.long, device=device), warm
            )
            adapter.crop(warm, base_warm)
    _sync(device)

    for width in widths:
        cache = adapter.new_cache()
        with torch.inference_mode():
            adapter.forward(torch.tensor([prompt], dtype=torch.long, device=device), cache)
            base = adapter.cache_length(cache)
            samples: list[int] = []
            block = torch.tensor([[1] * width], dtype=torch.long, device=device)
            for index in range(repeats + warmups):
                _sync(device)
                started = time.perf_counter_ns()
                adapter.forward(block, cache)
                _sync(device)
                elapsed = time.perf_counter_ns() - started
                adapter.crop(cache, base)
                if index >= warmups:
                    samples.append(elapsed)
        results[str(width)] = {
            "median_ns": statistics.median(samples),
            "min_ns": min(samples),
            "max_ns": max(samples),
            "count": len(samples),
        }
    return results


def write_profile(document: dict[str, Any], out: Path) -> None:
    """Persist a profile atomically, labelled as diagnostic."""
    provenance = collect_source_provenance()
    payload = {
        "kind": "profile",
        "data_kind": "measured",
        "is_benchmark_result": False,
        "disclaimer": (
            "Diagnostic per-stage timings with a synchronization between every "
            "stage. The added synchronizations inflate the total, so these "
            "numbers are for attribution and cost modelling only and are never "
            "mixed into a headline latency result."
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "commit": provenance.commit,
            "dirty": provenance.dirty,
            "source_sha256": provenance.source_sha256,
        },
        **document,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, out)


def format_profile(document: dict[str, Any]) -> str:
    """Readable summary for the terminal."""
    lines = ["  forward call cost by width (warm cache, synchronized)"]
    lines.append("    width   median_us")
    for width, values in sorted(
        document.get("forward_by_width", {}).items(), key=lambda item: int(item[0])
    ):
        lines.append(f"    {int(width):>5}   {values['median_ns'] / 1000:>9.1f}")
    lines.append("")
    lines.append("  speculative block stages (median us per block)")
    lines.append("    gamma    draft   verify   commit  catchup    whole   accept%  us/token")
    for entry in document.get("blocks", []):
        stages = entry["stages"]
        whole = stages["whole_block"]["median_ns"]
        rate = entry.get("acceptance_rate")
        tokens = entry.get("committed_tokens") or 1
        blocks = entry.get("blocks_measured") or 1
        per_token = whole * blocks / tokens / 1000
        lines.append(
            f"    {entry['gamma']:>5}"
            f"   {stages['draft_proposals']['median_ns'] / 1000:>6.0f}"
            f"   {stages['target_verification']['median_ns'] / 1000:>6.0f}"
            f"   {stages['commit_and_crop']['median_ns'] / 1000:>6.0f}"
            f"   {stages['draft_catchup']['median_ns'] / 1000:>6.0f}"
            f"   {whole / 1000:>6.0f}"
            f"   {100 * rate if rate is not None else 0:>7.1f}"
            f"   {per_token:>7.0f}"
        )
    lines.append("")
    lines.append("  Diagnostic pass: a synchronization between every stage inflates the total.")
    return "\n".join(lines)
