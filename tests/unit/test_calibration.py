"""Calibration fitting, profile freezing, and the summaries built on them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from switchback.calibration import (
    MAX_POSITIONS,
    fit_acceptance,
    format_profile,
    load_profile,
    write_profile,
)
from switchback.conformance import ConformanceCase
from switchback.conformance import summarize as conformance_summary
from switchback.controller import AcceptanceTable, CostProfile, CostTable
from switchback.controller_check import EngineRequest
from switchback.controller_check import summarize as engine_summary
from switchback.types import ConfigError


def block(accepted: int, rejection: int | None, action: str = "speculative") -> dict:
    return {
        "action": action,
        "accepted": accepted,
        "proposed": 8,
        "rejection_position": rejection,
    }


# --- acceptance fitting ----------------------------------------------------


def test_acceptance_counts_successes_and_the_single_failure() -> None:
    table = fit_acceptance([block(accepted=3, rejection=3)])
    assert table.successes[:4] == (1, 1, 1, 0)
    assert table.trials[:4] == (1, 1, 1, 1)


def test_positions_beyond_the_rejection_are_censored() -> None:
    table = fit_acceptance([block(accepted=2, rejection=2)])
    assert table.trials[3:] == (0,) * (MAX_POSITIONS - 3)
    assert table.successes[3:] == (0,) * (MAX_POSITIONS - 3)


def test_a_fully_accepted_block_has_no_failure() -> None:
    table = fit_acceptance([block(accepted=8, rejection=None)])
    assert table.successes == (1,) * MAX_POSITIONS
    assert table.trials == (1,) * MAX_POSITIONS


def test_non_speculative_blocks_are_ignored() -> None:
    table = fit_acceptance(
        [
            block(accepted=0, rejection=None, action="prefill"),
            block(accepted=0, rejection=None, action="target_only"),
        ]
    )
    assert table.trials == (0,) * MAX_POSITIONS


def test_counts_accumulate_across_blocks() -> None:
    table = fit_acceptance([block(3, 3), block(1, 1), block(8, None)])
    assert table.successes[0] == 3
    assert table.trials[0] == 3
    assert table.successes[1] == 2
    assert table.trials[1] == 3
    # Only the fully accepted block ever reached position 5.
    assert table.trials[5] == 1


def test_a_censored_table_does_not_report_zero_acceptance_for_long_positions() -> None:
    table = fit_acceptance([block(0, 0) for _ in range(30)])
    assert table.rate(0) < 0.05
    assert table.rate(6) == 0.5  # never evaluated; falls back to the prior


# --- profile freezing ------------------------------------------------------


def profile() -> CostProfile:
    return CostProfile(
        acceptance=AcceptanceTable((9,) * 8, (10,) * 8),
        cost=CostTable(
            target_forward_ns={1: 52_000_000.0, 2: 47_000_000.0, 9: 49_000_000.0},
            draft_forward_ns=12_700_000.0,
            block_overhead_ns=1_600_000.0,
            draft_prefill_ns=14_300_000.0,
            median_remaining_tokens=64.0,
        ),
        calibration_source="test",
    )


def test_a_profile_round_trips_with_its_hash(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    digest = write_profile(profile(), {"prompts": 4}, path, notes="test")
    restored = load_profile(path)
    assert restored.sha256() == digest
    assert restored.cost.draft_forward_ns == 12_700_000.0
    assert not list(tmp_path.glob("*.tmp"))


def test_an_edited_profile_fails_its_recorded_hash(tmp_path: Path) -> None:
    """Freezing the hash is what makes "calibrated before evaluation" checkable."""
    path = tmp_path / "calibration.json"
    write_profile(profile(), {}, path)
    document = json.loads(path.read_text())
    document["profile"]["margin"] = 0.0
    path.write_text(json.dumps(document))
    with pytest.raises(ConfigError, match="hash mismatch"):
        load_profile(path)


def test_an_unknown_calibration_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    write_profile(profile(), {}, path)
    document = json.loads(path.read_text())
    document["schema_version"] = 99
    path.write_text(json.dumps(document))
    with pytest.raises(ConfigError, match="schema_version"):
        load_profile(path)


def test_the_written_file_is_labelled_as_not_a_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    write_profile(profile(), {}, path)
    document = json.loads(path.read_text())
    assert document["is_benchmark_result"] is False
    assert "not latency results" in document["disclaimer"]


def test_the_formatted_profile_shows_the_acceptance_curve() -> None:
    text = format_profile(profile())
    assert "profile sha256" in text
    assert "conditional acceptance by position" in text
    assert "target forward by width" in text


# --- conformance summary ---------------------------------------------------


def case(length: int, identical: bool, gap: float | None = None) -> ConformanceCase:
    return ConformanceCase(
        engine="fixed_4",
        prompt_index=0,
        max_new_tokens=length,
        identical=identical,
        matched_prefix=length if identical else 10,
        first_divergence=None if identical else 10,
        divergence_margin=gap,
        divergence_logit_scale=40.0 if gap is not None else None,
        reference_token=None if identical else 1,
        candidate_token=None if identical else 2,
        reference_logit=None if gap is None else 39.25,
        candidate_logit=None if gap is None else 39.25 - gap,
        chosen_token_gap=gap,
    )


def test_conformance_summary_reports_agreement_per_length() -> None:
    summary = conformance_summary(
        [case(32, True), case(32, True), case(64, True), case(64, False, 0.25)]
    )
    assert summary["by_length"][32]["agreement_rate"] == 1.0
    assert summary["by_length"][64]["agreement_rate"] == 0.5
    assert summary["by_length"][64]["max_gap_between_chosen_tokens"] == 0.25
    assert summary["total_diverged"] == 1
    assert summary["largest_gap_between_chosen_tokens"] == 0.25


def test_conformance_summary_handles_perfect_agreement() -> None:
    summary = conformance_summary([case(32, True), case(32, True)])
    assert summary["total_diverged"] == 0
    assert summary["largest_gap_between_chosen_tokens"] is None
    assert summary["by_length"][32]["max_gap_between_chosen_tokens"] is None


# --- engine summary --------------------------------------------------------


def request(engine: str, latency_ms: float, **overrides) -> EngineRequest:
    base = dict(
        engine=engine,
        prompt_index=0,
        repeat=0,
        prompt_tokens=40,
        output_tokens=96,
        latency_ns=int(latency_ms * 1e6),
        ttft_ns=46_000_000,
        target_calls=24,
        draft_calls=100,
        proposed=80,
        accepted=64,
        controller_decisions=20,
        bypass_decisions=0,
        actions_chosen=[4] * 20,
    )
    base.update(overrides)
    return EngineRequest(**base)


def test_engine_summary_reports_acceptance_and_bypass_fractions() -> None:
    summary = engine_summary(
        [request("adaptive", 2500.0), request("adaptive", 2600.0, bypass_decisions=10)]
    )["adaptive"]
    assert summary["requests"] == 2
    assert summary["median_latency_ms"] == pytest.approx(2550.0)
    assert summary["acceptance"] == pytest.approx(0.8)
    assert summary["bypass_fraction"] == pytest.approx(0.25)
    assert summary["actions_chosen"] == {4: 40}


def test_engine_summary_reports_null_for_an_engine_that_never_drafts() -> None:
    summary = engine_summary(
        [
            request(
                "native_ar",
                5000.0,
                proposed=0,
                accepted=0,
                controller_decisions=0,
                draft_calls=0,
                actions_chosen=[],
            )
        ]
    )["native_ar"]
    assert summary["acceptance"] is None
    assert summary["bypass_fraction"] is None
    assert summary["actions_chosen"] == {}
