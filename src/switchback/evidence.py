"""Validation evidence: which checks ran, what they returned, what was measured.

The report renderer requires an ``evidence.json`` whose ``passed``,
``oracle_passed``, ``cache_passed`` and ``numerical_passed`` flags are true
before it will render a run (SPEC.md section 9.6). This module produces that
file by **actually executing** the named commands and recording their exit
codes, rather than by asserting the flags.

That distinction is the whole point, and it is still not a guarantee: the
renderer checks provenance, it cannot prove the acquisition was honest. What
this module adds is that a false flag requires editing a recorded exit code,
not just a boolean.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchback.provenance import collect_source_provenance, repo_root

EVIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckResult:
    """One executed validation command."""

    name: str
    category: str
    command: list[str]
    returncode: int
    duration_s: float
    passed: bool
    tail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckSpec:
    """A command to run, and which evidence flag it contributes to."""

    name: str
    category: str
    command: list[str]


def default_checks(python: str, include_gpu: bool) -> list[CheckSpec]:
    """The checks that back each evidence flag.

    ``oracle`` covers the exact-enumeration gates, ``cache`` the pending-token
    and rollback gates, ``numerical`` the transforms and the real-model
    differentials. GPU checks are included only when asked for: a CPU run cannot
    satisfy them, and their absence is recorded rather than assumed away.
    """
    checks = [
        CheckSpec(
            "oracle_enumeration",
            "oracle",
            [python, "-m", "pytest", "tests/unit/test_oracle.py", "-q"],
        ),
        CheckSpec(
            "sampled_speculation_enumeration",
            "oracle",
            [python, "-m", "pytest", "tests/unit/test_sampled_speculation.py", "-q"],
        ),
        CheckSpec(
            "cache_ledger",
            "cache",
            [python, "-m", "pytest", "tests/unit/test_cache.py", "-q"],
        ),
        CheckSpec(
            "speculation_paths",
            "cache",
            [python, "-m", "pytest", "tests/unit/test_speculation_paths.py", "-q"],
        ),
        CheckSpec(
            "cached_engine_cpu",
            "cache",
            [python, "-m", "pytest", "tests/integration/test_cached_engine_cpu.py", "-q"],
        ),
        CheckSpec(
            "sampling_transforms",
            "numerical",
            [python, "-m", "pytest", "tests/unit/test_sampling.py", "-q"],
        ),
        CheckSpec(
            "generated_properties",
            "numerical",
            [python, "-m", "pytest", "tests/property", "-q"],
        ),
        CheckSpec(
            "report_integrity",
            "report",
            [python, "-m", "pytest", "tests/report", "-q"],
        ),
    ]
    if include_gpu:
        checks.extend(
            [
                CheckSpec(
                    "cached_engine_gpu",
                    "numerical",
                    [
                        python,
                        "-m",
                        "pytest",
                        "tests/integration/test_cached_engine_gpu.py",
                        "-q",
                        "-m",
                        "gpu",
                    ],
                ),
                CheckSpec(
                    "speculation_gpu",
                    "cache",
                    [
                        python,
                        "-m",
                        "pytest",
                        "tests/integration/test_speculation_gpu.py",
                        "-q",
                        "-m",
                        "gpu",
                    ],
                ),
            ]
        )
    return checks


def run_check(spec: CheckSpec, cwd: Path, timeout_s: int = 3600) -> CheckResult:
    """Execute one check and record its exit code and output tail."""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            spec.command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        returncode = completed.returncode
        output = (completed.stdout + completed.stderr).strip()
    except subprocess.TimeoutExpired:
        returncode = -1
        output = f"timed out after {timeout_s}s"
    except OSError as error:
        returncode = -2
        output = f"{type(error).__name__}: {error}"
    return CheckResult(
        name=spec.name,
        category=spec.category,
        command=list(spec.command),
        returncode=returncode,
        duration_s=round(time.perf_counter() - started, 3),
        passed=returncode == 0,
        tail="\n".join(output.splitlines()[-6:]),
    )


@dataclass
class MeasuredFacts:
    """Numerical facts worth recording alongside the pass/fail flags.

    These are values this project actually measured, with the test that
    establishes each one. They are not thresholds chosen to make something pass.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, value: Any, units: str | None, source: str, note: str) -> None:
        self.entries.append(
            {"name": name, "value": value, "units": units, "source": source, "note": note}
        )


def measured_facts() -> MeasuredFacts:
    facts = MeasuredFacts()
    facts.add(
        "one_step_identity_error",
        0,
        "exact rational",
        "tests/unit/test_oracle.py::test_accepted_plus_residual_mass_equals_the_target_exactly",
        "Accepted plus residual mass equals the target exactly for every rational "
        "pair with denominator 4 over vocabularies of size 2, 3 and 4.",
    )
    facts.add(
        "sampled_decoder_total_variation_distance",
        0,
        "exact rational",
        "tests/unit/test_sampled_speculation.py::"
        "test_sampled_speculation_reproduces_the_target_distribution",
        "The sampled decoder driven through its cache, softmax, crop and catch-up "
        "reproduces the target sequence distribution exactly. Exactness is "
        "possible because every scripted row is uniform over a subset of size 1, "
        "2 or 4, which FP32 softmax represents without error.",
    )
    facts.add(
        "fp32_cache_logit_tolerance",
        1e-4,
        "absolute and relative",
        "tests/integration/test_cached_engine_cpu.py::"
        "test_cached_logits_match_full_prefix_recomputation",
        "Cached versus full-prefix logits on the FP32 tiny fixture, with exact "
        "argmax agreement at chunk widths 1, 2 and 5.",
    )
    facts.add(
        "bf16_cache_relative_logit_difference",
        0.021,
        "relative to max logit magnitude",
        "tests/integration/test_cached_engine_gpu.py::"
        "test_cached_logits_match_full_prefix_recomputation",
        "Measured on Qwen3-4B in BF16 on a GB10. Max absolute difference 0.73 to "
        "1.13 against a logit scale near 50. Argmax flips on one or two rows in "
        "thirty, always at near-ties with a top-2 margin of at most 0.125.",
    )
    facts.add(
        "greedy_conformance",
        "identical token ids",
        None,
        "tests/integration/test_speculation_gpu.py::"
        "test_fixed_speculation_is_token_identical_to_target_only",
        "Fixed greedy speculation at gamma 1, 2, 4 and 8 matches target-only "
        "decoding and hf_ar token for token at two context lengths.",
    )
    return facts


def build_evidence(
    results: list[CheckResult],
    include_gpu: bool,
    artifacts: list[str],
    notes: str = "",
) -> dict[str, Any]:
    """Assemble the evidence document from executed checks."""
    provenance = collect_source_provenance()

    def category_passed(category: str) -> bool:
        relevant = [result for result in results if result.category == category]
        return bool(relevant) and all(result.passed for result in relevant)

    oracle = category_passed("oracle")
    cache = category_passed("cache")
    numerical = category_passed("numerical")
    everything = all(result.passed for result in results) and bool(results)

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "passed": everything,
        "oracle_passed": oracle,
        "cache_passed": cache,
        "numerical_passed": numerical,
        "gpu_checks_included": include_gpu,
        "source_commit": provenance.commit,
        "source_dirty": provenance.dirty,
        "source_sha256": provenance.source_sha256,
        "commands": [" ".join(result.command) for result in results],
        "artifacts": artifacts,
        "checks": [result.as_dict() for result in results],
        "measured_facts": measured_facts().entries,
        "limits": (
            "Flags record the exit codes of commands that were executed. They do "
            "not prove the acquisition code is honest, and finite-model checks do "
            "not establish equivalence for every GPU kernel. A run without GPU "
            "checks has gpu_checks_included=false and cannot satisfy a GPU gate."
        ),
        "notes": notes,
    }


def write_evidence(document: dict[str, Any], out: Path) -> None:
    """Write the evidence document atomically."""
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, out)


def collect_evidence(
    python: str, include_gpu: bool, artifacts: list[str], notes: str = ""
) -> dict[str, Any]:
    """Run every default check and build the evidence document."""
    root = repo_root()
    results = [run_check(spec, root) for spec in default_checks(python, include_gpu)]
    return build_evidence(results, include_gpu, artifacts, notes)


def format_evidence(document: dict[str, Any]) -> str:
    """Readable summary for the terminal."""
    lines = ["  check                              result   seconds"]
    for check in document["checks"]:
        mark = "PASS" if check["passed"] else f"FAIL({check['returncode']})"
        lines.append(f"  {check['name']:<34} {mark:<8} {check['duration_s']:>7.1f}")
    lines.append("")
    for flag in ("passed", "oracle_passed", "cache_passed", "numerical_passed"):
        lines.append(f"  {flag:<20} {document[flag]}")
    lines.append(f"  {'gpu_checks_included':<20} {document['gpu_checks_included']}")
    return "\n".join(lines)
