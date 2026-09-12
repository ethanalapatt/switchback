"""Greedy conformance: where speculative and target-only output stop agreeing.

Greedy speculation is exact in real arithmetic. In BF16 it is not, and this
module measures exactly how not.

The mechanism is the one recorded in ``docs/correctness.md`` section 4a. A
target-only step computes its logits with a width-1 forward; a verification step
computes the same logits inside a width-``g+1`` forward. Different reduction
order, 8 mantissa bits, 36 layers. When the top two logits are within that noise,
the argmax flips, the two engines commit different tokens, and everything after
diverges.

So a divergence is reported together with the **top-2 margin at the position
where it happened**. A near-tie is floating point. A divergence at a confident
position would be a bug, and the two are not interchangeable.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchback.decoder import (
    EngineOptions,
    decode_speculative_greedy,
    decode_target_only,
)
from switchback.models.qwen import QwenAdapter
from switchback.provenance import collect_source_provenance
from switchback.sampling import forbidden_token_ids
from switchback.types import DecodeConfig


@dataclass(frozen=True)
class ConformanceCase:
    """One engine compared against target-only decoding on one prompt."""

    engine: str
    prompt_index: int
    max_new_tokens: int
    identical: bool
    matched_prefix: int
    first_divergence: int | None
    divergence_margin: float | None
    divergence_logit_scale: float | None
    reference_token: int | None
    candidate_token: int | None
    reference_logit: float | None = None
    candidate_logit: float | None = None
    chosen_token_gap: float | None = None


@dataclass(frozen=True)
class DivergenceDetail:
    """Uncached logits at the position where two engines disagreed."""

    margin: float
    logit_scale: float
    reference_logit: float
    candidate_logit: float


def _divergence_detail(
    target: QwenAdapter,
    prefix: Sequence[int],
    forbidden_ids: Sequence[int],
    reference_token: int,
    candidate_token: int,
    device: str,
) -> DivergenceDetail:
    """Recompute the row uncached and describe the disagreement there.

    The EOS mask the engines used is applied first. Without it the reported
    top-2 margin can be the gap between a suppressed stop token and everything
    else, which made a benign divergence read as a 15.5-logit blunder the first
    time this ran.

    ``reference_logit`` and ``candidate_logit`` are the two tokens the engines
    actually chose. That gap is the direct evidence: inside BF16 noise it is
    floating point, far outside it would be a bug.
    """
    import torch

    with torch.inference_mode():
        row = (
            target.model(
                input_ids=torch.tensor([list(prefix)], dtype=torch.long, device=device),
                use_cache=False,
            )
            .logits[0, -1]
            .float()
        )
    if forbidden_ids:
        row = row.clone()
        row.index_fill_(
            -1,
            torch.tensor(list(forbidden_ids), dtype=torch.long, device=row.device),
            float("-inf"),
        )
    finite = row[torch.isfinite(row)]
    top = row.topk(2)
    return DivergenceDetail(
        margin=float(top.values[0] - top.values[1]),
        logit_scale=float(finite.abs().max()),
        reference_logit=float(row[reference_token]),
        candidate_logit=float(row[candidate_token]),
    )


def check_prompt(
    target: QwenAdapter,
    draft: QwenAdapter,
    prompt_ids: Sequence[int],
    prompt_index: int,
    eos_token_ids: Sequence[int],
    max_new_tokens: int,
    device: str,
    gammas: Sequence[int] = (1, 2, 4, 8),
) -> list[ConformanceCase]:
    """Compare each fixed draft length against target-only on one prompt."""
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=max_new_tokens,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    options = EngineOptions(correctness_checks=True, device=device)
    forbidden = forbidden_token_ids(config, eos_token_ids)
    reference = decode_target_only(
        target, list(prompt_ids), config, list(eos_token_ids), options=options
    )
    cases: list[ConformanceCase] = []
    for gamma in gammas:
        candidate = decode_speculative_greedy(
            target,
            draft,
            list(prompt_ids),
            config,
            list(eos_token_ids),
            gamma=gamma,
            options=options,
        )
        matched = 0
        for left, right in zip(reference.output_ids, candidate.output_ids, strict=False):
            if left != right:
                break
            matched += 1
        identical = reference.output_ids == candidate.output_ids
        reference_token = None if identical else int(reference.output_ids[matched])
        candidate_token = None if identical else int(candidate.output_ids[matched])
        detail: DivergenceDetail | None = None
        if reference_token is not None and candidate_token is not None:
            detail = _divergence_detail(
                target,
                [*prompt_ids, *reference.output_ids[:matched]],
                forbidden,
                reference_token,
                candidate_token,
                device,
            )
        cases.append(
            ConformanceCase(
                engine=f"fixed_{gamma}",
                prompt_index=prompt_index,
                max_new_tokens=max_new_tokens,
                identical=identical,
                matched_prefix=matched,
                first_divergence=None if identical else matched,
                divergence_margin=None if detail is None else detail.margin,
                divergence_logit_scale=None if detail is None else detail.logit_scale,
                reference_token=reference_token,
                candidate_token=candidate_token,
                reference_logit=None if detail is None else detail.reference_logit,
                candidate_logit=None if detail is None else detail.candidate_logit,
                chosen_token_gap=(
                    None if detail is None else abs(detail.reference_logit - detail.candidate_logit)
                ),
            )
        )
    return cases


def summarize(cases: Sequence[ConformanceCase]) -> dict[str, Any]:
    """Agreement rates and the margins at which agreement broke."""
    by_length: dict[int, dict[str, Any]] = {}
    for length in sorted({case.max_new_tokens for case in cases}):
        sample = [case for case in cases if case.max_new_tokens == length]
        diverged = [case for case in sample if not case.identical]
        gaps = [case.chosen_token_gap for case in diverged if case.chosen_token_gap is not None]
        by_length[length] = {
            "comparisons": len(sample),
            "identical": len(sample) - len(diverged),
            "diverged": len(diverged),
            "agreement_rate": (len(sample) - len(diverged)) / len(sample) if sample else None,
            "max_gap_between_chosen_tokens": max(gaps) if gaps else None,
            "median_matched_prefix": (
                sorted(case.matched_prefix for case in diverged)[len(diverged) // 2]
                if diverged
                else None
            ),
        }
    gaps = [case.chosen_token_gap for case in cases if case.chosen_token_gap is not None]
    return {
        "by_length": by_length,
        "total_comparisons": len(cases),
        "total_diverged": sum(1 for case in cases if not case.identical),
        "largest_gap_between_chosen_tokens": max(gaps) if gaps else None,
    }


def write_conformance(
    out: Path,
    cases: Sequence[ConformanceCase],
    condition: dict[str, Any],
    notes: str,
) -> None:
    """Persist the conformance measurement atomically."""
    provenance = collect_source_provenance()
    document = {
        "kind": "greedy_conformance",
        "data_kind": "measured",
        "is_benchmark_result": False,
        "disclaimer": (
            "Measures where BF16 speculative and target-only decoding stop "
            "producing identical token ids, and the top-2 logit margin at each "
            "divergence. A divergence at a near-tie is floating point; a "
            "divergence at a confident position would be a bug."
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "condition": condition,
        "summary": summarize(cases),
        "cases": [asdict(case) for case in cases],
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


def format_conformance(cases: Sequence[ConformanceCase]) -> str:
    """Readable summary for the terminal."""
    summary = summarize(cases)
    lines = ["  length   comparisons   identical   agreement   max logit gap between choices"]
    for length, values in summary["by_length"].items():
        margin = values["max_gap_between_chosen_tokens"]
        lines.append(
            f"  {length:>6}   {values['comparisons']:>11}   {values['identical']:>9}"
            f"   {100 * (values['agreement_rate'] or 0):>8.1f}%"
            f"   {margin if margin is not None else 0:>26.3f}"
        )
    lines.append("")
    for case in cases:
        if case.identical:
            continue
        lines.append(
            f"  diverged: {case.engine} prompt {case.prompt_index} at token "
            f"{case.first_divergence}/{case.max_new_tokens}: chose "
            f"{case.candidate_token} (logit {case.candidate_logit:.3f}) where "
            f"target-only chose {case.reference_token} (logit "
            f"{case.reference_logit:.3f}); gap {case.chosen_token_gap:.3f} on a "
            f"logit scale of {case.divergence_logit_scale:.1f}"
        )
    return "\n".join(lines)
