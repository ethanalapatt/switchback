"""Evidence flags must follow the exit codes of commands that actually ran."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from switchback.evidence import (
    CheckResult,
    CheckSpec,
    build_evidence,
    default_checks,
    format_evidence,
    measured_facts,
    run_check,
    write_evidence,
)


def result(name: str, category: str, passed: bool) -> CheckResult:
    return CheckResult(
        name=name,
        category=category,
        command=["true" if passed else "false"],
        returncode=0 if passed else 1,
        duration_s=0.1,
        passed=passed,
        tail="",
    )


def test_all_flags_true_when_every_check_passes() -> None:
    results = [
        result("a", "oracle", True),
        result("b", "cache", True),
        result("c", "numerical", True),
        result("d", "report", True),
    ]
    document = build_evidence(results, include_gpu=False, artifacts=["x"])
    assert document["passed"]
    assert document["oracle_passed"]
    assert document["cache_passed"]
    assert document["numerical_passed"]
    assert document["gpu_checks_included"] is False


def test_a_failing_check_clears_its_category_and_the_overall_flag() -> None:
    results = [
        result("a", "oracle", True),
        result("b", "cache", False),
        result("c", "numerical", True),
    ]
    document = build_evidence(results, include_gpu=False, artifacts=["x"])
    assert document["cache_passed"] is False
    assert document["passed"] is False
    # Independent categories are not dragged down with it.
    assert document["oracle_passed"] is True
    assert document["numerical_passed"] is True


def test_a_category_with_no_checks_does_not_pass_by_default() -> None:
    """An absent check is not a passing check."""
    document = build_evidence([result("a", "oracle", True)], include_gpu=False, artifacts=["x"])
    assert document["oracle_passed"] is True
    assert document["cache_passed"] is False
    assert document["numerical_passed"] is False


def test_no_results_means_nothing_passed() -> None:
    document = build_evidence([], include_gpu=False, artifacts=["x"])
    assert document["passed"] is False


def test_commands_are_recorded_verbatim() -> None:
    results = [
        CheckResult("a", "oracle", ["python", "-m", "pytest", "tests/unit"], 0, 1.0, True, "")
    ]
    document = build_evidence(results, include_gpu=False, artifacts=["x"])
    assert document["commands"] == ["python -m pytest tests/unit"]
    assert document["checks"][0]["returncode"] == 0


def test_gpu_inclusion_is_recorded_and_adds_checks() -> None:
    without = {spec.name for spec in default_checks(sys.executable, include_gpu=False)}
    with_gpu = {spec.name for spec in default_checks(sys.executable, include_gpu=True)}
    assert with_gpu > without
    assert "speculation_gpu" in with_gpu
    assert "speculation_gpu" not in without
    document = build_evidence([result("a", "oracle", True)], include_gpu=True, artifacts=["x"])
    assert document["gpu_checks_included"] is True


def test_a_real_command_is_executed_and_its_exit_code_recorded(tmp_path: Path) -> None:
    passing = run_check(CheckSpec("ok", "oracle", [sys.executable, "-c", "pass"]), tmp_path)
    assert passing.passed and passing.returncode == 0
    failing = run_check(
        CheckSpec("bad", "oracle", [sys.executable, "-c", "raise SystemExit(3)"]), tmp_path
    )
    assert not failing.passed and failing.returncode == 3


def test_a_missing_executable_fails_rather_than_being_skipped(tmp_path: Path) -> None:
    missing = run_check(CheckSpec("nope", "oracle", ["definitely-not-a-real-binary"]), tmp_path)
    assert not missing.passed
    assert missing.returncode == -2


def test_the_output_tail_is_captured_for_a_failure(tmp_path: Path) -> None:
    check = run_check(
        CheckSpec(
            "noisy",
            "oracle",
            [sys.executable, "-c", "print('boom'); raise SystemExit(1)"],
        ),
        tmp_path,
    )
    assert "boom" in check.tail


def test_measured_facts_cite_the_test_that_established_them() -> None:
    for entry in measured_facts().entries:
        assert entry["source"], entry
        assert "::" in entry["source"] or entry["source"].endswith(".py")
        assert entry["note"]


def test_evidence_is_written_atomically_as_json(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "evidence.json"
    document = build_evidence([result("a", "oracle", True)], False, ["artifacts/x"])
    write_evidence(document, out)
    assert json.loads(out.read_text())["commands"] == ["true"]
    assert not list(out.parent.glob("*.tmp"))


def test_the_limits_note_is_present() -> None:
    document = build_evidence([result("a", "oracle", True)], False, ["x"])
    assert "cannot satisfy a GPU gate" in document["limits"]
    assert "honest" in document["limits"]


def test_formatted_evidence_shows_each_check() -> None:
    document = build_evidence(
        [result("a", "oracle", True), result("b", "cache", False)], False, ["x"]
    )
    text = format_evidence(document)
    assert "a " in text
    assert "FAIL(1)" in text
    assert "gpu_checks_included" in text
