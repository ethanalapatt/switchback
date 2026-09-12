"""Fit the controller's cost and acceptance tables from measured runs.

Calibration data only (SPEC.md section 4.4). Nothing here may touch the
held-out cohort, and the profile is hashed so a reviewer can confirm it was
frozen before evaluation.

Two quantities are fitted separately because they come from different kinds of
measurement:

* **Costs** come from a microbenchmark of forward calls at several widths, run
  from a warm cache with a synchronization around each call. Attribution needs
  those synchronizations; the resulting numbers are inputs to a model, not
  latency results.
* **Acceptance** comes from running fixed-length speculation over the
  calibration prompts and counting, per candidate position, how often a
  proposal survived verification -- censoring every position after the first
  rejection.
"""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchback.controller import (
    AcceptanceTable,
    CostProfile,
    CostTable,
    profile_from_dict,
)
from switchback.decoder import EngineOptions, decode_speculative_greedy
from switchback.events import ListSink
from switchback.models.qwen import QwenAdapter
from switchback.profiling import profile_single_forward
from switchback.provenance import collect_source_provenance
from switchback.types import ConfigError, DecodeConfig

CALIBRATION_SCHEMA_VERSION = 1

# Widths worth measuring: one per action plus its verification width.
CALIBRATION_WIDTHS: tuple[int, ...] = (1, 2, 3, 5, 9)
# The longest draft length in the action set decides how many candidate
# positions acceptance is fitted for.
MAX_POSITIONS = 8


def fit_acceptance(blocks: Sequence[dict[str, Any]]) -> AcceptanceTable:
    """Count conditional acceptance per position, censoring after a rejection.

    A block that accepted ``a`` of ``p`` proposals contributes a success at
    every position below ``a``, a single failure at ``a`` when a rejection
    occurred, and nothing at all above it. Positions above were never evaluated,
    so counting them as failures would bias long draft lengths downward exactly
    where the controller has to decide about them.
    """
    successes = [0] * MAX_POSITIONS
    trials = [0] * MAX_POSITIONS
    for block in blocks:
        if block.get("action") != "speculative":
            continue
        accepted = int(block["accepted"])
        for position in range(min(accepted, MAX_POSITIONS)):
            successes[position] += 1
            trials[position] += 1
        rejection = block["rejection_position"]
        if rejection is not None and rejection < MAX_POSITIONS:
            trials[rejection] += 1
    return AcceptanceTable(successes=tuple(successes), trials=tuple(trials))


def calibrate(
    target: QwenAdapter,
    draft: QwenAdapter,
    prompts: Sequence[Sequence[int]],
    eos_token_ids: Sequence[int],
    max_new_tokens: int,
    device: str,
    gamma: int = 8,
    repeats: int = 1,
) -> tuple[CostProfile, dict[str, Any]]:
    """Measure costs and acceptance, then assemble a frozen profile.

    ``gamma`` is the longest draft length so every candidate position gets
    observations. Using a shorter one would leave the long positions to the
    Beta(1, 1) prior, which is honest but uninformative.
    """
    if not prompts:
        raise ConfigError("calibration needs at least one prompt")

    target_costs = profile_single_forward(
        target, list(prompts[0]), widths=CALIBRATION_WIDTHS, device=device
    )
    draft_costs = profile_single_forward(draft, list(prompts[0]), widths=(1,), device=device)

    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=max_new_tokens,
        eos_policy="respect",
        seed=42,
    )
    options = EngineOptions(correctness_checks=False, device=device)
    blocks: list[dict[str, Any]] = []
    completion_lengths: list[int] = []
    block_overheads: list[float] = []

    for _ in range(repeats):
        for prompt in prompts:
            sink = ListSink()
            result = decode_speculative_greedy(
                target,
                draft,
                list(prompt),
                config,
                list(eos_token_ids),
                gamma=gamma,
                sink=sink,
                options=options,
            )
            blocks.extend(event.as_dict() for event in sink.events)
            completion_lengths.append(len(result.output_ids))
            for event in sink.events:
                if event.action != "speculative":
                    continue
                modelled = (
                    event.gamma * draft_costs["1"]["median_ns"]
                    + target_costs[str(event.gamma + 1)]["median_ns"]
                    if str(event.gamma + 1) in target_costs
                    else None
                )
                if modelled is not None:
                    block_overheads.append(max(0.0, event.duration_ns - modelled))

    # Draft prefill is measured as the cost of a forward over a whole prompt.
    prefill_costs = profile_single_forward(
        draft, [int(prompts[0][0])], widths=(len(prompts[0]),), device=device, repeats=8
    )
    profile = CostProfile(
        acceptance=fit_acceptance(blocks),
        cost=CostTable(
            target_forward_ns={
                int(width): float(values["median_ns"]) for width, values in target_costs.items()
            },
            draft_forward_ns=float(draft_costs["1"]["median_ns"]),
            block_overhead_ns=float(statistics.median(block_overheads)) if block_overheads else 0.0,
            draft_prefill_ns=float(next(iter(prefill_costs.values()))["median_ns"]),
            median_remaining_tokens=float(statistics.median(completion_lengths)),
        ),
        calibration_source=(
            f"{len(prompts)} calibration prompts x {repeats} repeat(s), "
            f"gamma={gamma}, max_new_tokens={max_new_tokens}"
        ),
        source_commit=collect_source_provenance().commit,
    )
    diagnostics = {
        "prompts": len(prompts),
        "repeats": repeats,
        "gamma": gamma,
        "blocks_observed": len(blocks),
        "completion_lengths": completion_lengths,
        "target_forward_ns": target_costs,
        "draft_forward_ns": draft_costs,
        "draft_prefill_ns": prefill_costs,
    }
    return profile, diagnostics


def write_profile(
    profile: CostProfile, diagnostics: dict[str, Any], out: Path, notes: str = ""
) -> str:
    """Persist the profile with its hash. Returns the hash."""
    digest = profile.sha256()
    document = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": "calibration",
        "data_kind": "measured",
        "is_benchmark_result": False,
        "disclaimer": (
            "Cost tables fitted from calibration prompts only. The microbenchmark "
            "synchronizes around every forward call, so these are model inputs, "
            "not latency results."
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile_sha256": digest,
        "profile": profile.as_dict(),
        "diagnostics": diagnostics,
        "notes": notes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, out)
    return digest


def load_profile(path: Path) -> CostProfile:
    """Read a calibration file and verify its recorded hash still matches."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ConfigError(
            f"calibration schema_version {document.get('schema_version')!r} is not "
            f"{CALIBRATION_SCHEMA_VERSION}"
        )
    profile = profile_from_dict(document["profile"])
    recorded = document.get("profile_sha256")
    if profile.sha256() != recorded:
        raise ConfigError(
            f"calibration profile hash mismatch: file records {recorded}, "
            f"content hashes to {profile.sha256()}"
        )
    return profile


def format_profile(profile: CostProfile) -> str:
    """Readable summary of a fitted profile."""
    lines = [
        f"  profile sha256   {profile.sha256()}",
        f"  source           {profile.calibration_source}",
        f"  margin           {profile.margin}  ewma_alpha {profile.ewma_alpha}",
        "",
        "  target forward by width (median us)",
    ]
    for width in sorted(profile.cost.target_forward_ns):
        lines.append(f"    {width:>3}  {profile.cost.target_forward_ns[width] / 1000:>9.1f}")
    lines += [
        f"  draft forward    {profile.cost.draft_forward_ns / 1000:.1f} us",
        f"  draft prefill    {profile.cost.draft_prefill_ns / 1000:.1f} us",
        f"  block overhead   {profile.cost.block_overhead_ns / 1000:.1f} us",
        f"  median length    {profile.cost.median_remaining_tokens:.0f} tokens",
        "",
        "  conditional acceptance by position",
    ]
    for position in range(profile.acceptance.positions):
        wins = profile.acceptance.successes[position]
        tries = profile.acceptance.trials[position]
        lines.append(
            f"    {position}  {profile.acceptance.rate(position):.3f}  ({wins}/{tries} observed)"
        )
    return "\n".join(lines)
