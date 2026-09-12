"""Traces must be replayable offline, and the replay must actually catch damage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from switchback.events import BlockEvent, ListSink
from switchback.traces import (
    TraceError,
    block_summary,
    build_header,
    format_replay,
    read_trace,
    replay_trace,
    write_trace,
)
from switchback.types import DecodeResult

PROMPT = [10, 11, 12, 13]


def block(
    block_id: int,
    action: str = "speculative",
    gamma: int = 2,
    proposed: int = 2,
    accepted: int = 2,
    rejection: int | None = None,
    committed: int = 3,
    target_cache: int = 0,
    draft_cache: int | None = 0,
    terminal: bool = False,
) -> BlockEvent:
    return BlockEvent(
        request_id="r",
        block_id=block_id,
        action=action,  # type: ignore[arg-type]
        gamma=gamma,
        proposed=proposed,
        accepted=accepted,
        rejection_position=rejection,
        committed=committed,
        target_calls=block_id + 1,
        draft_calls=block_id,
        target_cache_after=target_cache,
        draft_cache_after=draft_cache,
        bypass_reason=None,
        duration_ns=1000,
        terminal=terminal,
    )


def consistent_events() -> ListSink:
    """Prefill then two full blocks, with cache lengths that add up."""
    sink = ListSink()
    committed = 0
    committed += 1
    sink.events.append(
        block(
            0,
            action="prefill",
            gamma=0,
            proposed=0,
            accepted=0,
            committed=1,
            target_cache=len(PROMPT) + committed - 1,
            draft_cache=None,
        )
    )
    for index in (1, 2):
        committed += 3
        sink.events.append(
            block(
                index,
                accepted=2,
                proposed=2,
                committed=3,
                target_cache=len(PROMPT) + committed - 1,
                draft_cache=len(PROMPT) + committed - 1,
            )
        )
    return sink


def result_for(sink: ListSink) -> DecodeResult:
    committed = sum(event.committed for event in sink.events)
    return DecodeResult(
        run_id="run",
        request_id="r",
        output_ids=tuple(range(100, 100 + committed)),
        token_release_ns=tuple(range(1, committed + 1)),
        termination="budget",
        start_ns=0,
        first_token_ns=1,
        last_token_ns=committed,
        end_ns=committed + 1,
        proposed=sum(event.proposed for event in sink.events),
        accepted=sum(event.accepted for event in sink.events),
        target_calls=3,
        draft_calls=4,
        controller_decisions=0,
        bypass_decisions=0,
    )


def saved(tmp_path: Path, sink: ListSink) -> Path:
    result = result_for(sink)
    header = build_header(
        engine="fixed_2",
        result=result,
        prompt_ids=PROMPT,
        mode="greedy",
        eos_policy="respect",
        temperature=1.0,
        seed=42,
        gamma=2,
        max_new_tokens=16,
    )
    path = tmp_path / "trace.jsonl"
    write_trace(path, header, sink)
    return path


def test_a_consistent_trace_replays(tmp_path: Path) -> None:
    header, blocks = read_trace(saved(tmp_path, consistent_events()))
    report = replay_trace(header, blocks)
    assert report.ok, report.problems
    assert report.blocks == 3
    assert report.committed_tokens == 7
    assert report.acceptance_rate == 1.0


def test_the_prompt_text_is_not_stored_only_its_hash(tmp_path: Path) -> None:
    path = saved(tmp_path, consistent_events())
    header, _ = read_trace(path)
    # A trace should be shareable without redistributing dataset text, so the
    # prompt appears only as a length and a hash. (Checking for the digits of a
    # prompt token in the raw line would be a weaker test, not a stronger one:
    # "10" also occurs inside the output ids.)
    assert "prompt_ids" not in header
    assert "prompt_text" not in header
    assert len(header["prompt_sha256"]) == 64
    assert header["prompt_tokens"] == len(PROMPT)
    from switchback.provenance import sha256_bytes

    assert header["prompt_sha256"] == sha256_bytes(json.dumps(PROMPT).encode())


def test_a_wrong_cache_length_is_caught(tmp_path: Path) -> None:
    sink = consistent_events()
    sink.events[1] = block(
        1,
        accepted=2,
        proposed=2,
        committed=3,
        target_cache=sink.events[1].target_cache_after + 1,
        draft_cache=sink.events[1].draft_cache_after,
    )
    header, blocks = read_trace(saved(tmp_path, sink))
    report = replay_trace(header, blocks)
    assert not report.ok
    assert any("target cache" in problem for problem in report.problems)


def test_a_commit_count_inconsistent_with_acceptance_is_caught(tmp_path: Path) -> None:
    sink = consistent_events()
    event = sink.events[2]
    sink.events[2] = block(
        2,
        accepted=2,
        proposed=2,
        committed=2,
        target_cache=event.target_cache_after,
        draft_cache=event.draft_cache_after,
    )
    header, blocks = read_trace(saved(tmp_path, sink))
    report = replay_trace(header, blocks)
    assert not report.ok
    assert any("committed 2 for 2 accepted" in problem for problem in report.problems)


def test_a_rejection_position_that_disagrees_with_acceptance_is_caught(tmp_path: Path) -> None:
    sink = consistent_events()
    event = sink.events[1]
    sink.events[1] = block(
        1,
        accepted=2,
        proposed=2,
        committed=3,
        rejection=1,
        target_cache=event.target_cache_after,
        draft_cache=event.draft_cache_after,
    )
    header, blocks = read_trace(saved(tmp_path, sink))
    assert any("rejection at 1" in problem for problem in replay_trace(header, blocks).problems)


def test_accepted_beyond_proposed_is_caught(tmp_path: Path) -> None:
    sink = consistent_events()
    event = sink.events[1]
    sink.events[1] = block(
        1,
        accepted=5,
        proposed=2,
        committed=6,
        target_cache=event.target_cache_after,
        draft_cache=event.draft_cache_after,
    )
    header, blocks = read_trace(saved(tmp_path, sink))
    assert any("exceeds proposed" in problem for problem in replay_trace(header, blocks).problems)


def test_out_of_order_block_ids_are_caught(tmp_path: Path) -> None:
    sink = consistent_events()
    event = sink.events[2]
    sink.events[2] = block(
        7,
        accepted=2,
        proposed=2,
        committed=3,
        target_cache=event.target_cache_after,
        draft_cache=event.draft_cache_after,
    )
    header, blocks = read_trace(saved(tmp_path, sink))
    assert any("block_id is 7" in problem for problem in replay_trace(header, blocks).problems)


def test_a_terminal_block_may_commit_fewer_tokens(tmp_path: Path) -> None:
    """Ending on a committed EOS truncates the block, which is not damage."""
    sink = ListSink()
    sink.events.append(
        block(
            0,
            action="prefill",
            gamma=0,
            proposed=0,
            accepted=0,
            committed=1,
            target_cache=len(PROMPT),
            draft_cache=None,
        )
    )
    # Accepted 2 but committed only 1: the second emitted token was the EOS's
    # successor and was discarded. The draft cache is one short because the
    # catch-up call was skipped.
    sink.events.append(
        block(
            1,
            accepted=2,
            proposed=2,
            committed=1,
            target_cache=len(PROMPT) + 1,
            draft_cache=len(PROMPT),
            terminal=True,
        )
    )
    header, blocks = read_trace(saved(tmp_path, sink))
    report = replay_trace(header, blocks)
    assert report.ok, report.problems


def test_a_terminal_block_still_has_to_commit_something(tmp_path: Path) -> None:
    sink = ListSink()
    sink.events.append(
        block(
            0,
            action="prefill",
            gamma=0,
            proposed=0,
            accepted=0,
            committed=1,
            target_cache=len(PROMPT),
            draft_cache=None,
        )
    )
    sink.events.append(
        block(
            1,
            accepted=2,
            proposed=2,
            committed=0,
            target_cache=len(PROMPT),
            draft_cache=len(PROMPT),
            terminal=True,
        )
    )
    header, blocks = read_trace(saved(tmp_path, sink))
    report = replay_trace(header, blocks)
    assert not report.ok
    assert any("terminal block committed 0" in problem for problem in report.problems)


def test_a_terminal_draft_cache_two_short_is_still_wrong(tmp_path: Path) -> None:
    """The relaxation is exactly one position, not an unbounded exemption."""
    sink = ListSink()
    sink.events.append(
        block(
            0,
            action="prefill",
            gamma=0,
            proposed=0,
            accepted=0,
            committed=1,
            target_cache=len(PROMPT),
            draft_cache=None,
        )
    )
    sink.events.append(
        block(
            1,
            accepted=2,
            proposed=2,
            committed=1,
            target_cache=len(PROMPT) + 1,
            draft_cache=len(PROMPT) - 1,
            terminal=True,
        )
    )
    header, blocks = read_trace(saved(tmp_path, sink))
    assert any("draft cache" in problem for problem in replay_trace(header, blocks).problems)


def test_header_and_block_counter_disagreement_is_caught(tmp_path: Path) -> None:
    path = saved(tmp_path, consistent_events())
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["accepted"] = 99
    path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    header, blocks = read_trace(path)
    assert any("header says" in problem for problem in replay_trace(header, blocks).problems)


def test_a_malformed_trace_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("")
    with pytest.raises(TraceError, match="empty"):
        read_trace(path)
    path.write_text(json.dumps({"record": "block"}) + "\n")
    with pytest.raises(TraceError, match="header"):
        read_trace(path)
    path.write_text(json.dumps({"record": "header", "schema_version": 99}) + "\n")
    with pytest.raises(TraceError, match="schema_version"):
        read_trace(path)


def test_block_summary_counts_rejections_by_position(tmp_path: Path) -> None:
    sink = consistent_events()
    event = sink.events[2]
    sink.events[2] = block(
        2,
        accepted=1,
        proposed=2,
        committed=2,
        rejection=1,
        target_cache=event.target_cache_after - 1,
        draft_cache=event.draft_cache_after - 1,
    )
    _, blocks = read_trace(saved(tmp_path, sink))
    summary = block_summary(blocks)
    assert summary["speculative_blocks"] == 2
    assert summary["rejected_blocks"] == 1
    assert summary["fully_accepted_blocks"] == 1
    assert summary["rejections_by_position"] == {1: 1}


def test_formatted_replay_states_the_verdict(tmp_path: Path) -> None:
    header, blocks = read_trace(saved(tmp_path, consistent_events()))
    text = format_replay(header, replay_trace(header, blocks))
    assert "replay PASSED" in text
    assert "acceptance" in text


def test_the_trace_is_written_atomically(tmp_path: Path) -> None:
    path = saved(tmp_path, consistent_events())
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
