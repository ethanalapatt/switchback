"""Milestone 6 check: the controller against the fixed baselines.

This is **not** the benchmark. SPEC.md M6 asks for the controller to be run
against fixed lengths on pilot data and for one correct decision and one
limitation to be documented. Three things disqualify it as a result:

* the prompts are the same ones the profile was calibrated on, so it is
  in-sample;
* there are four prompts, not a locked cohort;
* there are no confidence intervals.

What it can establish is that the controller runs, that every engine still
produces identical greedy tokens, and which action it picks and why.

Engine order is randomized within each repeat, and everything runs in one
process, because per-token cost on this machine varies by roughly ten percent
between processes. Cross-process latency comparisons are not trustworthy here.
"""

from __future__ import annotations

import json
import os
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchback.controller import CostController, CostProfile, FixedController
from switchback.decoder import (
    EngineOptions,
    decode_greedy_with_controller,
    decode_target_only,
)
from switchback.events import ListSink
from switchback.models.qwen import QwenAdapter
from switchback.provenance import collect_source_provenance
from switchback.types import DecodeConfig, DecodeResult

SCHEDULING_SEED = 2026


@dataclass(frozen=True)
class EngineRequest:
    """One measured request. Field names carry their units."""

    engine: str
    prompt_index: int
    repeat: int
    prompt_tokens: int
    output_tokens: int
    latency_ns: int
    ttft_ns: int
    target_calls: int
    draft_calls: int
    proposed: int
    accepted: int
    controller_decisions: int
    bypass_decisions: int
    actions_chosen: list[int]


def _engines(
    target: QwenAdapter,
    draft: QwenAdapter,
    profile: CostProfile,
    config: DecodeConfig,
    eos_token_ids: Sequence[int],
    options: EngineOptions,
) -> dict[str, Any]:
    """Every engine under test, each a callable taking prompt ids and a sink."""

    def native(ids: list[int], sink: ListSink) -> DecodeResult:
        return decode_target_only(
            target, ids, config, list(eos_token_ids), sink=sink, options=options
        )

    def fixed(gamma: int) -> Any:
        def run(ids: list[int], sink: ListSink) -> DecodeResult:
            return decode_greedy_with_controller(
                target,
                draft,
                ids,
                config,
                list(eos_token_ids),
                FixedController(gamma),
                sink=sink,
                options=options,
            )

        return run

    def adaptive(allow_bypass: bool) -> Any:
        def run(ids: list[int], sink: ListSink) -> DecodeResult:
            controller = CostController(profile=profile, allow_bypass=allow_bypass)
            controller.reset()
            return decode_greedy_with_controller(
                target,
                draft,
                ids,
                config,
                list(eos_token_ids),
                controller,
                sink=sink,
                options=options,
            )

        return run

    return {
        "native_ar": native,
        "fixed_1": fixed(1),
        "fixed_2": fixed(2),
        "fixed_4": fixed(4),
        "fixed_8": fixed(8),
        "adaptive": adaptive(True),
        "adaptive_no_bypass": adaptive(False),
    }


def run_check(
    target: QwenAdapter,
    draft: QwenAdapter,
    profile: CostProfile,
    prompts: Sequence[Sequence[int]],
    eos_token_ids: Sequence[int],
    max_new_tokens: int,
    device: str,
    repeats: int = 3,
    warmups: int = 2,
) -> tuple[list[EngineRequest], list[str]]:
    """Run every engine over every prompt, with randomized engine order."""
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=max_new_tokens,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    options = EngineOptions(correctness_checks=True, device=device)
    engines = _engines(target, draft, profile, config, eos_token_ids, options)
    rng = random.Random(SCHEDULING_SEED)

    for _ in range(warmups):
        for run in engines.values():
            run(list(prompts[0]), ListSink())

    measured: list[EngineRequest] = []
    outputs: dict[tuple[int, str], tuple[int, ...]] = {}
    for repeat in range(repeats):
        for index, prompt in enumerate(prompts):
            order = list(engines)
            rng.shuffle(order)
            for name in order:
                sink = ListSink()
                result = engines[name](list(prompt), sink)
                outputs.setdefault((index, name), result.output_ids)
                measured.append(
                    EngineRequest(
                        engine=name,
                        prompt_index=index,
                        repeat=repeat,
                        prompt_tokens=len(prompt),
                        output_tokens=len(result.output_ids),
                        latency_ns=result.end_ns - result.start_ns,
                        ttft_ns=result.first_token_ns - result.start_ns,
                        target_calls=result.target_calls,
                        draft_calls=result.draft_calls,
                        proposed=result.proposed,
                        accepted=result.accepted,
                        controller_decisions=result.controller_decisions,
                        bypass_decisions=result.bypass_decisions,
                        actions_chosen=[
                            event.gamma for event in sink.events if event.action == "speculative"
                        ],
                    )
                )

    mismatches = [
        f"prompt {index}: {name} differs from native_ar"
        for index in range(len(prompts))
        for name in engines
        if outputs.get((index, name)) != outputs.get((index, "native_ar"))
    ]
    return measured, mismatches


def summarize(requests: Sequence[EngineRequest]) -> dict[str, dict[str, Any]]:
    """Per-engine medians and counters. Descriptive only."""
    summary: dict[str, dict[str, Any]] = {}
    for engine in sorted({request.engine for request in requests}):
        sample = [request for request in requests if request.engine == engine]
        proposed = sum(request.proposed for request in sample)
        accepted = sum(request.accepted for request in sample)
        decisions = sum(request.controller_decisions for request in sample)
        bypasses = sum(request.bypass_decisions for request in sample)
        actions: dict[int, int] = {}
        for request in sample:
            for action in request.actions_chosen:
                actions[action] = actions.get(action, 0) + 1
        summary[engine] = {
            "requests": len(sample),
            "median_latency_ms": statistics.median(r.latency_ns / 1e6 for r in sample),
            "median_ttft_ms": statistics.median(r.ttft_ns / 1e6 for r in sample),
            "median_target_calls": statistics.median(r.target_calls for r in sample),
            "median_draft_calls": statistics.median(r.draft_calls for r in sample),
            "acceptance": (accepted / proposed) if proposed else None,
            "bypass_fraction": (bypasses / decisions) if decisions else None,
            "actions_chosen": dict(sorted(actions.items())),
        }
    return summary


def write_check(
    out: Path,
    requests: Sequence[EngineRequest],
    mismatches: Sequence[str],
    profile: CostProfile,
    condition: dict[str, Any],
    notes: str,
) -> None:
    """Persist the check atomically, labelled as in-sample and not a benchmark."""
    provenance = collect_source_provenance()
    document = {
        "kind": "controller_check",
        "data_kind": "measured",
        "is_benchmark_result": False,
        "disclaimer": (
            "In-sample: these prompts are the ones the cost profile was "
            "calibrated on. Four prompts, no held-out cohort, no confidence "
            "intervals. Establishes that the controller runs and that greedy "
            "output is unchanged; it is not evidence of a speedup."
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile_sha256": profile.sha256(),
        "scheduling_seed": SCHEDULING_SEED,
        "condition": condition,
        "greedy_output_mismatches": list(mismatches),
        "summary": summarize(requests),
        "requests": [asdict(request) for request in requests],
        "source": {
            "commit": provenance.commit,
            "dirty": provenance.dirty,
            "source_sha256": provenance.source_sha256,
        },
        "notes": notes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, out)


def format_check(summary: dict[str, dict[str, Any]]) -> str:
    """Readable table for the terminal."""
    lines = [
        "  engine                n   median_ms   target_calls  draft_calls"
        "  accept%  bypass%  actions",
        "  " + "-" * 100,
    ]
    order = [
        "native_ar",
        "fixed_1",
        "fixed_2",
        "fixed_4",
        "fixed_8",
        "adaptive",
        "adaptive_no_bypass",
    ]
    for engine in order:
        values = summary.get(engine)
        if values is None:
            continue
        acceptance = values["acceptance"]
        bypass = values["bypass_fraction"]
        lines.append(
            f"  {engine:<20} {values['requests']:>2}"
            f"   {values['median_latency_ms']:>9.1f}"
            f"   {values['median_target_calls']:>12.0f}"
            f"  {values['median_draft_calls']:>11.0f}"
            f"  {(100 * acceptance) if acceptance is not None else 0:>7.1f}"
            f"  {(100 * bypass) if bypass is not None else 0:>7.1f}"
            f"  {values['actions_chosen']}"
        )
    lines.append("")
    lines.append("  In-sample controller check on calibration prompts. Not a benchmark result.")
    return "\n".join(lines)
