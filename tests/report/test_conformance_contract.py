"""The renderer's near-tie contract for greedy mismatches (ADR 0004).

Greedy speculation is exact in real arithmetic, so a mismatch is either
floating point or a bug. These tests pin that the renderer admits the first
only with evidence, and never admits the second.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_results.py"
SPEC = importlib.util.spec_from_file_location("report_renderer", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)

BASELINE_OUTPUT = [11, 22, 33, 44]
DIVERGED_OUTPUT = [11, 22, 99, 77]


class ConformanceContractTests(unittest.TestCase):
    """All fixtures below use synthetic timings. None is benchmark evidence."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for prompt in range(2):
            for engine, duration in (("hf_ar", 300_000_000), ("adaptive", 200_000_000)):
                row = dict(
                    cohort="fixture",
                    dataset="synthetic_only",
                    mode="greedy",
                    condition="fixed_length_fixture",
                    prompt_id=f"test_{prompt}",
                    seed=42,
                    repeat=0,
                    engine=engine,
                    status="ok",
                    start_ns=1_000_000_000,
                    first_token_ns=1_010_000_000,
                    last_token_ns=1_000_000_000 + duration,
                    end_ns=1_000_000_000 + duration,
                    output_ids=list(BASELINE_OUTPUT),
                )
                row.update({name: None for name in renderer.COUNTERS})
                self.rows.append(row)
        self.evidence = dict(
            passed=True,
            oracle_passed=True,
            cache_passed=True,
            numerical_passed=True,
            source_commit="a" * 40,
            commands=["synthetic fixture only"],
            artifacts=["synthetic fixture only"],
        )
        self.manifest = dict(
            schema_version=1,
            data_kind="fixture",
            complete=True,
            git_commit="a" * 40,
            git_dirty=False,
            hardware={"kind": "fixture"},
            software={"kind": "fixture"},
            model_revisions={"kind": "fixture"},
            baseline="hf_ar",
            required_engines=["hf_ar", "adaptive"],
        )
        for name in (
            "config_sha256",
            "workload_sha256",
            "calibration_sha256",
            "benchmark_source_sha256",
        ):
            self.manifest[name] = "b" * 64

    def write(self):
        self.manifest["expected_keys"] = [list(renderer.key(row)) for row in self.rows]
        (self.root / "requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )
        (self.root / "evidence.json").write_text(json.dumps(self.evidence))
        self.manifest["files"] = {
            name: renderer.digest(self.root / name) for name in ("requests.jsonl", "evidence.json")
        }
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def diverge(self, gap=0.25, **overrides):
        """Make the adaptive row for prompt 0 differ, with evidence attached."""
        row = next(r for r in self.rows if r["engine"] == "adaptive")
        row["output_ids"] = list(DIVERGED_OUTPUT)
        divergence = dict(
            index=2,
            reference_token=BASELINE_OUTPUT[2],
            candidate_token=DIVERGED_OUTPUT[2],
            chosen_token_gap=gap,
            logit_scale=39.2,
        )
        divergence.update(overrides)
        row["greedy_divergence"] = divergence
        return row

    # --- the strict default -------------------------------------------------

    def test_without_a_declared_bound_any_mismatch_is_still_refused(self):
        """The original contract survives for runs that do not opt in."""
        self.diverge()
        self.write()
        with self.assertRaisesRegex(ValueError, "declares no max_chosen_token_gap"):
            renderer.load_run(self.root, True)

    def test_identical_outputs_need_no_evidence(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.write()
        renderer.load_run(self.root, True)

    # --- admitting a documented near-tie ------------------------------------

    def test_a_near_tie_within_the_bound_is_admitted(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=0.25)
        self.write()
        manifest, evidence, rows = renderer.load_run(self.root, True)
        report = renderer.render(manifest, evidence, rows)
        self.assertIn("Greedy conformance", report)
        self.assertIn("Greedy match", report)
        # One of two adaptive requests matched.
        self.assertIn("50.0%", report)

    def test_a_gap_beyond_the_bound_with_nothing_adjudicating_it_is_refused(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=15.5)
        self.write()
        with self.assertRaisesRegex(ValueError, "nothing adjudicated it"):
            renderer.load_run(self.root, True)

    # --- adjudication (ADR 0006) -------------------------------------------

    def adjudicate(self, verdict="near_tie", smallest=0.25, **overrides):
        """Attach an adjudication for the diverging adaptive row."""
        row = next(r for r in self.rows if r["engine"] == "adaptive")
        item = dict(
            {name: row[name] for name in renderer.KEY_FIELDS},
            index=2,
            reference_token=BASELINE_OUTPUT[2],
            candidate_token=DIVERGED_OUTPUT[2],
            screen_gap=15.5,
            smallest_margin=smallest,
            any_path_prefers_candidate=True,
            engine_replay_margin=-0.25,
            verdict=verdict,
            paths=[],
        )
        item.update(overrides)
        (self.root / "adjudication.jsonl").write_text(json.dumps(item) + "\n")
        self.manifest.setdefault("files", {})
        self.manifest["adjudication"] = {"adjudicated": 1, "unexplained": 0}
        return item

    def write_with_adjudication(self):
        self.write()
        self.manifest["files"]["adjudication.jsonl"] = renderer.digest(
            self.root / "adjudication.jsonl"
        )
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_an_adjudicated_near_tie_is_admitted(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=15.5)
        self.adjudicate()
        self.write_with_adjudication()
        report = renderer.render(*renderer.load_run(self.root, True))
        self.assertIn("adjudicated by recomputing", report)

    def test_an_unexplained_adjudication_is_still_refused(self):
        """Adjudication may explain a divergence. It may never suppress one."""
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=15.5)
        self.adjudicate(verdict="unexplained", smallest=9.0)
        self.write_with_adjudication()
        with self.assertRaisesRegex(ValueError, "adjudicated unexplained"):
            renderer.load_run(self.root, True)

    def test_a_near_tie_verdict_must_agree_with_its_own_margin(self):
        """A verdict that contradicts the number beside it is not evidence."""
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=15.5)
        self.adjudicate(verdict="near_tie", smallest=7.0)
        self.write_with_adjudication()
        with self.assertRaisesRegex(ValueError, "still exceeds the bound"):
            renderer.load_run(self.root, True)

    def test_an_adjudication_for_a_different_position_does_not_apply(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=15.5)
        self.adjudicate(index=0)
        self.write_with_adjudication()
        with self.assertRaisesRegex(ValueError, "nothing adjudicated it"):
            renderer.load_run(self.root, True)

    def test_an_adjudication_for_a_different_engine_does_not_apply(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=15.5)
        self.adjudicate(engine="hf_ar")
        self.write_with_adjudication()
        with self.assertRaisesRegex(ValueError, "nothing adjudicated it"):
            renderer.load_run(self.root, True)

    def test_a_tampered_adjudication_file_is_refused(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=15.5)
        self.adjudicate()
        self.write_with_adjudication()
        with (self.root / "adjudication.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Integrity mismatch for adjudication"):
            renderer.load_run(self.root, True)

    def test_a_missing_adjudication_file_is_refused(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=15.5)
        self.adjudicate()
        self.write_with_adjudication()
        (self.root / "adjudication.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "but it is missing"):
            renderer.load_run(self.root, True)

    def test_an_unknown_verdict_is_refused(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=15.5)
        self.adjudicate(verdict="probably fine")
        self.write_with_adjudication()
        with self.assertRaisesRegex(ValueError, "Unknown adjudication verdict"):
            renderer.load_run(self.root, True)

    def test_adjudication_cannot_rescue_a_mismatch_with_no_evidence(self):
        """The row must still carry its own divergence record."""
        self.manifest["max_chosen_token_gap"] = 0.5
        row = self.diverge(gap=15.5)
        del row["greedy_divergence"]
        self.adjudicate()
        self.write_with_adjudication()
        with self.assertRaisesRegex(ValueError, "no recorded divergence evidence"):
            renderer.load_run(self.root, True)

    def test_a_mismatch_without_evidence_is_refused(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        row = self.diverge()
        del row["greedy_divergence"]
        self.write()
        with self.assertRaisesRegex(ValueError, "no recorded divergence evidence"):
            renderer.load_run(self.root, True)

    def test_evidence_pointing_at_the_wrong_position_is_refused(self):
        """Evidence must describe the divergence that actually happened."""
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(index=0)
        self.write()
        with self.assertRaisesRegex(ValueError, "not the first actual difference"):
            renderer.load_run(self.root, True)

    def test_evidence_naming_the_wrong_tokens_is_refused(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(candidate_token=12345)
        self.write()
        with self.assertRaisesRegex(ValueError, "different tokens"):
            renderer.load_run(self.root, True)

    def test_evidence_without_a_usable_gap_is_refused(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(chosen_token_gap=None)
        self.write()
        with self.assertRaisesRegex(ValueError, "no usable chosen_token_gap"):
            renderer.load_run(self.root, True)
        self.diverge(chosen_token_gap=-1.0)
        self.write()
        with self.assertRaisesRegex(ValueError, "no usable chosen_token_gap"):
            renderer.load_run(self.root, True)

    def test_a_nonsense_bound_is_refused(self):
        self.manifest["max_chosen_token_gap"] = 0
        self.write()
        with self.assertRaisesRegex(ValueError, "positive number"):
            renderer.load_run(self.root, True)

    # --- what the report says about it --------------------------------------

    def test_the_report_states_the_bound_it_applied(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=0.25)
        self.write()
        report = renderer.render(*renderer.load_run(self.root, True))
        self.assertIn("near-tie bound of 0.5", report)
        self.assertIn("0004-bf16-greedy-conformance", report)

    def test_a_fully_conformant_run_reports_a_hundred_percent(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.write()
        report = renderer.render(*renderer.load_run(self.root, True))
        self.assertIn("100.0%", report)

    def test_conformance_is_not_reported_for_sampled_cohorts(self):
        for row in self.rows:
            row["mode"] = "sample"
        self.manifest["max_chosen_token_gap"] = 0.5
        self.write()
        report = renderer.render(*renderer.load_run(self.root, True))
        # Sampled outputs are not required to match, so the column is N/A.
        self.assertIn("N/A", report)
        self.assertNotIn("100.0%", report)

    def test_rendering_stays_deterministic(self):
        self.manifest["max_chosen_token_gap"] = 0.5
        self.diverge(gap=0.25)
        self.write()
        args = renderer.load_run(self.root, True)
        self.assertEqual(renderer.render(*args), renderer.render(*args))


if __name__ == "__main__":
    unittest.main()
