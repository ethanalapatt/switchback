"""The viewer's replay check must agree with the Python one.

viewer/viewer.js reimplements replay_trace on purpose: a viewer that trusted
the file it was handed could draw a confident picture of an inconsistent trace.
Two implementations are only worth having if they are kept in agreement, so
this drives the JavaScript one through node and compares verdicts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from switchback.events import BlockEvent, ListSink
from switchback.provenance import repo_root
from switchback.traces import build_header, read_trace, replay_trace, write_trace
from switchback.types import DecodeResult

VIEWER = repo_root() / "viewer"
NODE = shutil.which("node")

DRIVER = """
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const body = src.slice(src.indexOf('const TRACE_SCHEMA_VERSION'), src.indexOf('function stat('));
const ctx = { console };
vm.createContext(ctx);
vm.runInContext(body + '\\nthis.parseTrace = parseTrace; this.replay = replay;', ctx);
const parsed = ctx.parseTrace(fs.readFileSync(process.argv[3], 'utf8'));
const verdict = ctx.replay(parsed.header, parsed.blocks);
process.stdout.write(JSON.stringify({
  ok: verdict.ok, problems: verdict.problems,
  committed: verdict.committed, accepted: verdict.accepted, proposed: verdict.proposed,
}));
"""

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def javascript_replay(trace: Path, tmp_path: Path) -> dict:
    driver = tmp_path / "driver.js"
    driver.write_text(DRIVER)
    result = subprocess.run(
        [str(NODE), str(driver), str(VIEWER / "viewer.js"), str(trace)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def build_trace(path: Path, events: list[BlockEvent], prompt_tokens: int, outputs: int) -> None:
    sink = ListSink()
    sink.events.extend(events)
    result = DecodeResult(
        run_id="run",
        request_id="r",
        output_ids=tuple(range(100, 100 + outputs)),
        token_release_ns=tuple(range(1, outputs + 1)),
        termination="budget",
        start_ns=0,
        first_token_ns=1,
        last_token_ns=outputs,
        end_ns=outputs + 1,
        proposed=sum(event.proposed for event in events),
        accepted=sum(event.accepted for event in events),
        target_calls=3,
        draft_calls=4,
        controller_decisions=0,
        bypass_decisions=0,
    )
    header = build_header(
        engine="fixed_2",
        result=result,
        prompt_ids=list(range(prompt_tokens)),
        mode="greedy",
        eos_policy="respect",
        temperature=1.0,
        seed=42,
        gamma=2,
        max_new_tokens=16,
    )
    write_trace(path, header, sink)


def block(block_id, action, gamma, proposed, accepted, rejection, committed, target, draft):
    return BlockEvent(
        request_id="r",
        block_id=block_id,
        action=action,
        gamma=gamma,
        proposed=proposed,
        accepted=accepted,
        rejection_position=rejection,
        committed=committed,
        target_calls=block_id + 1,
        draft_calls=block_id,
        target_cache_after=target,
        draft_cache_after=draft,
        bypass_reason=None,
        duration_ns=1_000_000,
    )


def consistent(prompt: int = 4) -> list[BlockEvent]:
    return [
        block(0, "prefill", 0, 0, 0, None, 1, prompt, None),
        block(1, "speculative", 2, 2, 2, None, 3, prompt + 3, prompt + 3),
        block(2, "speculative", 2, 2, 1, 1, 2, prompt + 5, prompt + 5),
    ]


def test_the_viewer_exists_and_is_self_contained() -> None:
    """No network: the page must reference only files next to it."""
    html = (VIEWER / "index.html").read_text()
    assert 'src="viewer.js"' in html
    assert 'href="style.css"' in html
    assert "http://" not in html and "https://" not in html
    for name in ("viewer.js", "style.css", "example-trace.js"):
        assert (VIEWER / name).is_file(), name


def test_the_bundled_example_is_a_real_trace() -> None:
    text = (VIEWER / "example-trace.js").read_text()
    assert "SWITCHBACK_EXAMPLE_TRACE" in text
    payload = json.loads(text[text.index("= ") + 2 : text.rindex(";")])
    header = json.loads(payload.splitlines()[0])
    assert header["record"] == "header"
    # A measured trace, not a hand-drawn illustration.
    assert header["source"]["commit"]
    assert header["prompt_tokens"] > 0


@needs_node
@pytest.mark.parametrize("name", ["greedy_g4.jsonl", "sampled_g4.jsonl"])
def test_both_implementations_agree_on_a_real_trace(tmp_path: Path, name: str) -> None:
    trace = repo_root() / "artifacts" / "traces" / name
    if not trace.is_file():
        pytest.skip(f"{trace} not present")
    header, blocks = read_trace(trace)
    expected = replay_trace(header, blocks)
    actual = javascript_replay(trace, tmp_path)
    assert actual["ok"] is expected.ok is True
    assert actual["committed"] == expected.committed_tokens
    assert actual["accepted"] == expected.accepted
    assert actual["proposed"] == expected.proposed


@needs_node
def test_both_implementations_agree_that_a_good_trace_is_good(tmp_path: Path) -> None:
    trace = tmp_path / "good.jsonl"
    build_trace(trace, consistent(), prompt_tokens=4, outputs=6)
    header, blocks = read_trace(trace)
    assert replay_trace(header, blocks).ok
    assert javascript_replay(trace, tmp_path)["ok"] is True


@needs_node
@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        ("cache", "target cache"),
        ("commit", "committed"),
        ("order", "block_id"),
        ("accepted", "exceeds proposed"),
    ],
)
def test_both_implementations_reject_the_same_damage(
    tmp_path: Path, mutation: str, needle: str
) -> None:
    events = consistent()
    if mutation == "cache":
        events[1] = block(1, "speculative", 2, 2, 2, None, 3, 99, 7)
    elif mutation == "commit":
        events[2] = block(2, "speculative", 2, 2, 1, 1, 3, 10, 10)
    elif mutation == "order":
        events[2] = block(9, "speculative", 2, 2, 1, 1, 2, 9, 9)
    else:
        events[1] = block(1, "speculative", 2, 2, 5, None, 6, 10, 10)
    trace = tmp_path / "bad.jsonl"
    build_trace(trace, events, prompt_tokens=4, outputs=sum(e.committed for e in events))
    header, blocks = read_trace(trace)
    python_report = replay_trace(header, blocks)
    javascript = javascript_replay(trace, tmp_path)
    assert not python_report.ok
    assert javascript["ok"] is False
    assert any(needle in problem for problem in python_report.problems)
    assert any(needle in problem for problem in javascript["problems"])
