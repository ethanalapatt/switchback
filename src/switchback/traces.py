"""Compact block traces and an offline replay checker.

A trace is written **after** the timed region (SPEC.md section 9.4): the engine
collects events into an in-memory list during the request and this module
serializes them afterwards, so JSON encoding never lands inside a latency
measurement.

The replay checker re-derives the pending-token convention from the trace alone,
without a model or a GPU. That is what makes a saved trace evidence rather than
decoration: a reviewer can download one file and check that the cache lengths
are consistent with the tokens that were committed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchback.events import ListSink
from switchback.provenance import collect_source_provenance, sha256_bytes
from switchback.types import DecodeResult

TRACE_SCHEMA_VERSION = 1


class TraceError(ValueError):
    """A saved trace is internally inconsistent."""


@dataclass
class TraceHeader:
    """Identity of the request a trace describes."""

    schema_version: int
    engine: str
    run_id: str
    request_id: str
    mode: str
    eos_policy: str
    temperature: float
    seed: int
    gamma: int | None
    max_new_tokens: int
    prompt_tokens: int
    prompt_sha256: str
    output_ids: list[int]
    termination: str
    target_calls: int
    draft_calls: int
    proposed: int
    accepted: int
    models: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""


def build_header(
    engine: str,
    result: DecodeResult,
    prompt_ids: list[int],
    mode: str,
    eos_policy: str,
    temperature: float,
    seed: int,
    gamma: int | None,
    max_new_tokens: int,
    models: dict[str, Any] | None = None,
) -> TraceHeader:
    """Summarize a completed request for the head of its trace file."""
    provenance = collect_source_provenance()
    return TraceHeader(
        schema_version=TRACE_SCHEMA_VERSION,
        engine=engine,
        run_id=result.run_id,
        request_id=result.request_id,
        mode=mode,
        eos_policy=eos_policy,
        temperature=temperature,
        seed=seed,
        gamma=gamma,
        max_new_tokens=max_new_tokens,
        prompt_tokens=len(prompt_ids),
        # The prompt itself is not stored: a trace should be shareable without
        # redistributing dataset text. The hash still pins which prompt it was.
        prompt_sha256=sha256_bytes(json.dumps(prompt_ids).encode()),
        output_ids=list(result.output_ids),
        termination=result.termination,
        target_calls=result.target_calls,
        draft_calls=result.draft_calls,
        proposed=result.proposed,
        accepted=result.accepted,
        models=models or {},
        source={
            "commit": provenance.commit,
            "dirty": provenance.dirty,
            "source_sha256": provenance.source_sha256,
        },
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_trace(path: Path, header: TraceHeader, sink: ListSink) -> None:
    """Write a JSONL trace atomically: one header line, then one line per block."""
    lines = [json.dumps({"record": "header", **asdict(header)}, sort_keys=True)]
    lines.extend(
        json.dumps({"record": "block", **event.as_dict()}, sort_keys=True) for event in sink.events
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a JSONL trace back into its header and block records."""
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise TraceError(f"{path} is empty")
    header, *blocks = records
    if header.get("record") != "header":
        raise TraceError("first record must be the header")
    if header.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise TraceError(
            f"trace schema_version {header.get('schema_version')!r} is not {TRACE_SCHEMA_VERSION}"
        )
    for index, block in enumerate(blocks):
        if block.get("record") != "block":
            raise TraceError(f"record {index + 1} is not a block")
    return header, blocks


@dataclass(frozen=True)
class ReplayReport:
    """What the replay checker concluded from a trace, and why."""

    ok: bool
    blocks: int
    committed_tokens: int
    accepted: int
    proposed: int
    problems: tuple[str, ...] = ()

    @property
    def acceptance_rate(self) -> float | None:
        return self.accepted / self.proposed if self.proposed else None


def replay_trace(header: dict[str, Any], blocks: list[dict[str, Any]]) -> ReplayReport:
    """Re-derive the cache convention from the trace alone.

    Checks, in order: block ids are consecutive from zero; a rejection position
    equals the accepted count; accepted never exceeds proposed; a speculative
    block commits ``accepted + 1`` tokens; both cache lengths equal
    ``prompt + committed - 1`` at every boundary; and the committed tokens sum
    to the recorded output length.

    A block flagged ``terminal`` is held to relaxed but still specific rules,
    because ending on a committed EOS legitimately truncates the block and skips
    the draft catch-up. The exceptions are named rather than blanket: a terminal
    block must still commit between one and ``accepted + 1`` tokens, and its
    draft cache must be the expected length or exactly one less.
    """
    problems: list[str] = []
    prompt_tokens = int(header["prompt_tokens"])
    committed = 0
    accepted = 0
    proposed = 0

    for index, block in enumerate(blocks):
        label = f"block {index}"
        if block["block_id"] != index:
            problems.append(f"{label}: block_id is {block['block_id']}, expected {index}")
        rejection = block["rejection_position"]
        if rejection is not None and rejection != block["accepted"]:
            problems.append(f"{label}: rejection at {rejection} but {block['accepted']} accepted")
        if block["accepted"] > block["proposed"]:
            problems.append(
                f"{label}: accepted {block['accepted']} exceeds proposed {block['proposed']}"
            )
        terminal = bool(block.get("terminal", False))
        if block["action"] == "speculative":
            expected_commits = block["accepted"] + 1
            if terminal:
                # A block that committed an EOS discards the rest of its own
                # output, so it may commit fewer tokens than it accepted plus
                # one. It must still commit at least one.
                if not 1 <= block["committed"] <= expected_commits:
                    problems.append(
                        f"{label}: terminal block committed {block['committed']}, "
                        f"expected 1..{expected_commits}"
                    )
            elif block["committed"] != expected_commits:
                problems.append(
                    f"{label}: committed {block['committed']} for {block['accepted']} accepted"
                )
        committed += block["committed"]
        accepted += block["accepted"]
        proposed += block["proposed"]

        expected_length = prompt_tokens + committed - 1
        if block["target_cache_after"] != expected_length:
            problems.append(
                f"{label}: target cache is {block['target_cache_after']}, "
                f"expected {expected_length}"
            )
        draft_length = block["draft_cache_after"]
        if draft_length is not None:
            # On a terminal block the draft's catch-up call is skipped on
            # purpose, so its cache is allowed to be one position short.
            lower = expected_length - 1 if terminal else expected_length
            if not lower <= draft_length <= expected_length:
                problems.append(
                    f"{label}: draft cache is {draft_length}, expected "
                    + (f"{lower} or {expected_length}" if terminal else str(expected_length))
                )

    if committed != len(header["output_ids"]):
        problems.append(
            f"blocks committed {committed} tokens but the header records "
            f"{len(header['output_ids'])}"
        )
    if accepted != header["accepted"] or proposed != header["proposed"]:
        problems.append(
            f"block counters sum to {accepted}/{proposed}, header says "
            f"{header['accepted']}/{header['proposed']}"
        )
    return ReplayReport(
        ok=not problems,
        blocks=len(blocks),
        committed_tokens=committed,
        accepted=accepted,
        proposed=proposed,
        problems=tuple(problems),
    )


def format_replay(header: dict[str, Any], report: ReplayReport) -> str:
    """Human-readable replay summary."""
    rate = report.acceptance_rate
    lines = [
        f"  trace       {header['engine']} / {header['request_id']}",
        f"  mode        {header['mode']} temperature={header['temperature']} seed={header['seed']}",
        f"  gamma       {header['gamma']}",
        f"  prompt      {header['prompt_tokens']} tokens (sha256 {header['prompt_sha256'][:12]})",
        f"  output      {len(header['output_ids'])} tokens, terminated on {header['termination']}",
        f"  blocks      {report.blocks}",
        f"  calls       target={header['target_calls']} draft={header['draft_calls']}",
        f"  acceptance  {report.accepted}/{report.proposed}"
        + (f" ({100 * rate:.1f}%)" if rate is not None else " (n/a)"),
        f"  source      {header['source'].get('commit')}",
        "",
        f"  replay {'PASSED' if report.ok else 'FAILED'}: cache lengths are "
        f"{'consistent with' if report.ok else 'INCONSISTENT with'} the committed tokens",
    ]
    for problem in report.problems:
        lines.append(f"    {problem}")
    return "\n".join(lines)


def block_summary(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts a viewer or report can show without recomputation."""
    speculative = [block for block in blocks if block["action"] == "speculative"]
    rejections = [block for block in speculative if block["rejection_position"] is not None]
    by_position: dict[int, int] = {}
    for block in rejections:
        position = int(block["rejection_position"])
        by_position[position] = by_position.get(position, 0) + 1
    return {
        "blocks": len(blocks),
        "speculative_blocks": len(speculative),
        "rejected_blocks": len(rejections),
        "fully_accepted_blocks": len(speculative) - len(rejections),
        "rejections_by_position": dict(sorted(by_position.items())),
    }
