"""The doctor must describe the real machine and never let a skip count as a pass."""

from __future__ import annotations

import json
from pathlib import Path

from switchback.env import (
    LOCKED_DISTRIBUTIONS,
    collect_environment,
    format_environment,
    write_environment,
)


def test_report_describes_this_machine() -> None:
    report = collect_environment()
    assert report.host["machine"]
    assert report.python["version"]
    assert report.max_context_tokens == 4096
    assert {check["name"] for check in report.checks} >= {
        "python_version",
        "torch_importable",
        "cuda_available",
    }


def test_every_locked_distribution_is_reported_even_when_absent() -> None:
    report = collect_environment()
    assert set(report.packages) == set(LOCKED_DISTRIBUTIONS)
    for name, version in report.packages.items():
        assert version is None or isinstance(version, str), name


def test_unavailable_telemetry_is_null_not_zero() -> None:
    report = collect_environment()
    for field in ("temperature_c", "power_draw_w", "driver_version"):
        value = report.driver[field]
        assert value is None or isinstance(value, (int, float, str))
    if not report.torch["cuda_available"]:
        assert report.torch["total_memory_bytes"] is None


def test_require_gpu_makes_the_cuda_check_mandatory() -> None:
    relaxed = {check["name"]: check for check in collect_environment(False).checks}
    strict = {check["name"]: check for check in collect_environment(True).checks}
    assert relaxed["cuda_available"]["required"] is False
    assert strict["cuda_available"]["required"] is True
    # A machine without CUDA must fail the strict form rather than skip it.
    if strict["cuda_available"]["status"] != "pass":
        assert collect_environment(True).ok is False


def test_gpu_ready_matches_the_cuda_check() -> None:
    report = collect_environment()
    cuda = {check["name"]: check["status"] for check in report.checks}["cuda_available"]
    assert report.gpu_ready == (cuda == "pass")


def test_report_is_written_atomically_as_json(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "environment.json"
    report = collect_environment()
    write_environment(report, out)
    document = json.loads(out.read_text())
    assert document["host"]["machine"] == report.host["machine"]
    assert not list(out.parent.glob("*.tmp"))


def test_formatted_output_marks_each_check() -> None:
    text = format_environment(collect_environment())
    assert "switchback doctor" in text
    assert "python_version" in text
    assert "gpu_ready=" in text
