"""Integrity tests using synthetic timings only, never benchmark evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_results.py"
SPEC = importlib.util.spec_from_file_location("report_renderer", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


class RendererIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for prompt in range(3):
            for repeat in range(2):
                for engine, duration in (("hf_ar", 300_000_000), ("adaptive", 200_000_000)):
                    row = dict(cohort="fixture", dataset="synthetic_only", mode="greedy",
                               condition="fixed_length_fixture", prompt_id=f"test_{prompt}", seed=42,
                               repeat=repeat, engine=engine, status="ok", start_ns=1_000_000_000,
                               first_token_ns=1_010_000_000, last_token_ns=1_000_000_000 + duration,
                               end_ns=1_000_000_000 + duration, output_ids=[1, 2, 3])
                    row.update({name: None for name in renderer.COUNTERS})
                    self.rows.append(row)
        self.evidence = dict(passed=True, oracle_passed=True, cache_passed=True, numerical_passed=True,
                             source_commit="a" * 40, commands=["synthetic fixture only"],
                             artifacts=["synthetic fixture only"])
        self.manifest = dict(schema_version=1, data_kind="fixture", complete=True,
                             git_commit="a" * 40, git_dirty=False, hardware={"kind": "fixture"},
                             software={"kind": "fixture"}, model_revisions={"kind": "fixture"},
                             baseline="hf_ar", required_engines=["hf_ar", "adaptive"],
                             expected_keys=[list(renderer.key(row)) for row in self.rows])
        for name in ("config_sha256", "workload_sha256", "calibration_sha256", "benchmark_source_sha256"):
            self.manifest[name] = "b" * 64
        self.write()

    def write(self):
        (self.root / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        (self.root / "evidence.json").write_text(json.dumps(self.evidence))
        self.manifest["files"] = {name: renderer.digest(self.root / name)
                                  for name in ("requests.jsonl", "evidence.json")}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_valid_fixture_is_watermarked_and_deterministic(self):
        args = renderer.load_run(self.root, True)
        first = renderer.render(*args)
        self.assertIn("SYNTHETIC TEST FIXTURE", first)
        self.assertIn("1.500x", first)  # Known artificial ratio, not a measured achievement.
        self.assertEqual(first, renderer.render(*args))

    def test_fixture_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "allow-fixture"):
            renderer.load_run(self.root, False)

    def test_integrity_tampering_is_rejected(self):
        with (self.root / "requests.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Integrity mismatch"):
            renderer.load_run(self.root, True)

    def test_missing_request_is_not_silently_dropped(self):
        self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
            renderer.load_run(self.root, True)

    def test_duplicate_request_is_rejected(self):
        self.rows.append(self.rows[0].copy())
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate completed"):
            renderer.load_run(self.root, True)

    def test_greedy_output_mismatch_blocks_performance_claim(self):
        self.rows[1]["output_ids"] = [9, 8, 7]
        self.write()
        with self.assertRaisesRegex(ValueError, "Greedy output mismatch"):
            renderer.load_run(self.root, True)

    def test_failed_validation_blocks_report(self):
        self.evidence["cache_passed"] = False
        self.write()
        with self.assertRaisesRegex(ValueError, "cache_passed"):
            renderer.load_run(self.root, True)

    def test_impossible_timing_blocks_report(self):
        self.rows[0]["end_ns"] = 0
        self.write()
        with self.assertRaisesRegex(ValueError, "Incoherent timing"):
            renderer.load_run(self.root, True)

    def test_cli_refuses_fixture_results_filename(self):
        output = self.root / "RESULTS.md"
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.root), "--allow-fixture",
                                 "--out", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())
        self.assertIn("must never be named", result.stderr)

    def test_cli_writes_separate_fixture_report(self):
        output = self.root / "fixture_report.md"
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.root), "--allow-fixture",
                                 "--out", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SYNTHETIC TEST FIXTURE", output.read_text())


if __name__ == "__main__":
    unittest.main()
