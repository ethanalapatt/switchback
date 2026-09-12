#!/usr/bin/env python3
"""Render benchmark evidence, never generate benchmark measurements.

Usage: python scripts/render_results.py RUN_DIRECTORY --out RESULTS.md
Standard library only. See SPEC.md for the acquisition/provenance contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

KEY_FIELDS = ("cohort", "dataset", "mode", "condition", "prompt_id", "seed", "repeat", "engine")
COUNTERS = (
    "accepted",
    "proposed",
    "target_calls",
    "bypass_decisions",
    "controller_decisions",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def key(row):
    return tuple(row[name] for name in KEY_FIELDS)


def quantile(values, probability):
    ordered = sorted(values)
    require(bool(ordered), "Cannot calculate quantile of empty sample")
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def safe(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def fmt(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def summed(rows, field):
    values = [row[field] for row in rows]
    return None if any(value is None for value in values) else sum(values)


def ratio(numerator, denominator):
    return None if numerator is None or not denominator else numerator / denominator


def latency(row):
    return (row["end_ns"] - row["start_ns"]) / 1e9


def prompt_latencies(rows):
    grouped = defaultdict(list)
    for row in rows:
        # Preserve dataset identity when pooling the two real-workload strata.
        grouped[(row["dataset"], row["prompt_id"])].append(latency(row))
    return {prompt: statistics.median(values) for prompt, values in grouped.items()}


def paired_speedup(baseline, candidate):
    left, right = prompt_latencies(baseline), prompt_latencies(candidate)
    require(left.keys() == right.keys(), "Unpaired prompt populations")
    prompts = sorted(left)
    logs = [math.log(left[p] / right[p]) for p in prompts]
    point = math.exp(statistics.mean(logs))
    rng = random.Random(2026)
    draws = [math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(2000)]
    slow = sum(right[p] > 1.05 * left[p] for p in prompts) / len(prompts)
    return point, quantile(draws, 0.025), quantile(draws, 0.975), slow


def load_run(root, allow_fixture):
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest.get("schema_version") == 1, "Unsupported schema version")
    kind = manifest.get("data_kind")
    require(kind in {"measured", "fixture"}, "Unknown data_kind")
    require(kind == "measured" or allow_fixture, "Fixture input requires --allow-fixture")
    require(manifest.get("complete") is True, "Run is incomplete")
    require(manifest.get("git_dirty") is False, "Final run must have a clean source tree")
    require(
        re.fullmatch(r"[0-9a-f]{40}", manifest.get("git_commit", "")) is not None,
        "Missing immutable source commit",
    )
    for name in ("hardware", "software", "model_revisions"):
        require(isinstance(manifest.get(name), dict) and bool(manifest[name]), f"Missing {name}")
    for name in (
        "config_sha256",
        "workload_sha256",
        "calibration_sha256",
        "benchmark_source_sha256",
    ):
        require(
            re.fullmatch(r"[0-9a-f]{64}", manifest.get(name, "")) is not None, f"Missing {name}"
        )
    for filename in ("requests.jsonl", "evidence.json"):
        require(
            manifest.get("files", {}).get(filename) == digest(root / filename),
            f"Integrity mismatch for {filename}",
        )
    evidence = json.loads((root / "evidence.json").read_text())
    for name in ("passed", "oracle_passed", "cache_passed", "numerical_passed"):
        require(evidence.get(name) is True, f"Validation did not pass: {name}")
    require(evidence.get("source_commit") == manifest["git_commit"], "Evidence/source mismatch")
    require(
        isinstance(evidence.get("commands"), list) and evidence["commands"],
        "No validation commands",
    )
    require(
        isinstance(evidence.get("artifacts"), list) and evidence["artifacts"],
        "No validation artifact references",
    )
    rows = [
        json.loads(line)
        for line in (root / "requests.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(bool(rows), "No requests")
    expected = manifest.get("expected_keys")
    require(isinstance(expected, list) and bool(expected), "No expected coverage manifest")
    require(
        all(isinstance(item, list) and len(item) == len(KEY_FIELDS) for item in expected),
        "Malformed expected request key",
    )
    expected = [tuple(item) for item in expected]
    require(len(set(expected)) == len(expected), "Duplicate expected request key")
    observed = []
    for row in rows:
        current = key(row)
        observed.append(current)
        require(row.get("status") == "ok", f"Unsuccessful request {current}")
        for name in (*KEY_FIELDS[:5], "engine"):
            require(isinstance(row[name], str) and bool(row[name]), f"Invalid {name}")
        require(is_int(row["seed"]) and is_int(row["repeat"]), "Invalid seed/repeat")
        require(row["mode"] in {"greedy", "sample"}, "Unknown decoding mode")
        times = [row[name] for name in ("start_ns", "first_token_ns", "last_token_ns", "end_ns")]
        require(all(is_int(value) and value >= 0 for value in times), "Invalid timestamps")
        require(times[0] < times[1] <= times[2] <= times[3], "Incoherent timing boundaries")
        ids = row.get("output_ids")
        require(
            isinstance(ids, list) and bool(ids) and all(is_int(i) and i >= 0 for i in ids),
            "Missing or invalid committed token IDs",
        )
        if len(ids) == 1:
            require(times[1] == times[2], "Single-token release timestamps disagree")
        for name in COUNTERS:
            require(name in row, f"Missing counter {name}; use null if unavailable")
            require(row[name] is None or (is_int(row[name]) and row[name] >= 0), f"Invalid {name}")
        for part, total in (("accepted", "proposed"), ("bypass_decisions", "controller_decisions")):
            require(
                (row[part] is None) == (row[total] is None), f"Partial counter pair {part}/{total}"
            )
            if row[part] is not None:
                require(row[part] <= row[total], f"Impossible counter pair {part}/{total}")
        require(row["target_calls"] is None or row["target_calls"] > 0, "No target call")
        if row["peak_allocated_bytes"] is not None and row["peak_reserved_bytes"] is not None:
            require(
                row["peak_allocated_bytes"] <= row["peak_reserved_bytes"],
                "Allocated memory exceeds reserved",
            )
    require(len(observed) == len(set(observed)), "Duplicate completed request key")
    require(
        set(observed) == set(expected), "Missing or unexpected requests; report cannot drop rows"
    )
    engines = manifest.get("required_engines")
    require(
        isinstance(engines, list) and len(engines) >= 2 and len(engines) == len(set(engines)),
        "Invalid required_engines",
    )
    baseline = manifest.get("baseline")
    require(baseline in engines, "Baseline must be a required engine")
    require({row["engine"] for row in rows} == set(engines), "Required engine coverage mismatch")
    cells = defaultdict(dict)
    for row in rows:
        cells[key(row)[:-1]][row["engine"]] = row
    bound = manifest.get("max_chosen_token_gap")
    require(
        bound is None or (isinstance(bound, (int, float)) and bound > 0),
        "max_chosen_token_gap must be a positive number when present",
    )
    for cell, members in cells.items():
        require(set(members) == set(engines), f"Missing paired engines for {cell}")
        if cell[2] == "greedy":
            reference = members[baseline]["output_ids"]
            for engine, row in members.items():
                if row["output_ids"] == reference:
                    continue
                check_greedy_divergence(cell, engine, row, reference, bound)
    return manifest, evidence, rows


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right))


def check_greedy_divergence(cell, engine, row, reference, bound):
    """A greedy mismatch is admissible only as a documented near-tie.

    Greedy speculation is exact in real arithmetic, so a mismatch is either
    floating point or a bug. The two are distinguished by the logit gap between
    the tokens the engines actually chose: inside BF16 reduction-order noise it
    is floating point; outside it is not (ADR 0004).

    The runner records that gap. This function refuses a mismatch that carries
    no evidence, whose evidence points at a different position than the actual
    first difference, or whose gap exceeds the bound the manifest declared. A
    run that does not declare a bound keeps the original strict behaviour: any
    greedy mismatch is refused.
    """
    label = f"Greedy output mismatch: {cell}, {engine}"
    require(
        bound is not None,
        f"{label}; the manifest declares no max_chosen_token_gap, so no "
        f"mismatch is admissible. Investigate before publishing a speedup.",
    )
    divergence = row.get("greedy_divergence")
    require(
        isinstance(divergence, dict),
        f"{label}; no recorded divergence evidence. A mismatch without evidence "
        f"is a bug, not a near-tie.",
    )
    index = divergence.get("index")
    expected_index = first_difference(reference, row["output_ids"])
    require(
        is_int(index) and index == expected_index,
        f"{label}; recorded divergence index {index} is not the first actual "
        f"difference at {expected_index}.",
    )
    require(
        divergence.get("reference_token") == reference[expected_index]
        and divergence.get("candidate_token") == row["output_ids"][expected_index],
        f"{label}; recorded divergence names different tokens than the outputs do.",
    )
    gap = divergence.get("chosen_token_gap")
    require(
        isinstance(gap, (int, float)) and gap >= 0,
        f"{label}; recorded divergence has no usable chosen_token_gap.",
    )
    require(
        gap <= bound,
        f"{label}; the engines chose tokens {gap} logits apart, beyond the "
        f"declared near-tie bound of {bound}. That is not floating point.",
    )


def conformance_rate(rows, baseline_engine, cells):
    """Fraction of greedy cells where an engine matched the baseline exactly."""
    matched = 0
    total = 0
    for row in rows:
        members = cells.get(key(row)[:-1])
        if members is None or row["mode"] != "greedy":
            continue
        total += 1
        matched += row["output_ids"] == members[baseline_engine]["output_ids"]
    return (matched / total) if total else None


def render(manifest, evidence, rows):
    fixture = manifest["data_kind"] == "fixture"
    lines = ["# Switchback results", ""]
    if fixture:
        lines += ["**SYNTHETIC TEST FIXTURE. NOT BENCHMARK EVIDENCE.**", ""]
    lines += [
        "Generated from raw request measurements. Do not edit numerical values by hand.",
        "",
        f"Source commit: `{manifest['git_commit']}`. Baseline: `{safe(manifest['baseline'])}`.",
        "",
        "```json",
        json.dumps(
            {
                name: manifest[name]
                for name in (
                    "hardware",
                    "software",
                    "model_revisions",
                    "config_sha256",
                    "workload_sha256",
                    "calibration_sha256",
                    "benchmark_source_sha256",
                    "files",
                )
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        f"Complete logical requests: {len(rows)}. "
        f"Required engines: {len(manifest['required_engines'])}.",
        "",
        "Greedy token equality was checked for every paired request. "
        "Sampled outputs were not required to match.",
        "",
        "Oracle, cache, and numerical gates: passed according to the linked validation evidence. "
        "Finite-model tests do not prove universal GPU equivalence.",
        "",
        "Validation commands: " + "; ".join(safe(x) for x in evidence["commands"]) + ".",
        "",
        "Evidence references: " + "; ".join(safe(x) for x in evidence["artifacts"]) + ".",
        "",
    ]
    grouped = defaultdict(list)
    pooled = defaultdict(list)
    for row in rows:
        grouped[(row["cohort"], row["dataset"], row["mode"], row["condition"])].append(row)
        # Real cohorts and controlled stress cohorts must have distinct cohort names.
        pooled[
            (row["cohort"], "ALL DATASETS IN THIS COHORT", row["mode"], row["condition"])
        ].append(row)
    groups = {**grouped, **pooled}
    for group, population in sorted(groups.items()):
        by_engine = defaultdict(list)
        for row in population:
            by_engine[row["engine"]].append(row)
        reference = by_engine[manifest["baseline"]]
        lines += [
            "## " + " / ".join(safe(x) for x in group),
            "",
            "| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | "
            "TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for engine in manifest["required_engines"]:
            sample = by_engine[engine]
            durations = [latency(row) for row in sample]
            ttft = [(row["first_token_ns"] - row["start_ns"]) / 1e6 for row in sample]
            tpot = [
                (row["last_token_ns"] - row["first_token_ns"]) / 1e6 / (len(row["output_ids"]) - 1)
                for row in sample
                if len(row["output_ids"]) > 1
            ]
            speed, lo, hi, slowdown = paired_speedup(reference, sample)
            values = [
                safe(engine),
                str(len(sample)),
                str(len(prompt_latencies(sample))),
                fmt(statistics.median(durations) * 1000),
                fmt(quantile(durations, 0.95) * 1000),
                fmt(statistics.median(ttft)),
                fmt(statistics.median(tpot) if tpot else None),
                fmt(sum(len(row["output_ids"]) for row in sample) / sum(durations)),
                fmt(speed) + "x",
                f"[{lo:.3f}, {hi:.3f}]",
                f"{100 * slowdown:.1f}%",
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines += [
            "",
            "| Engine | Greedy match | Accepted/proposed | Output tokens/target call | "
            "Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for engine in manifest["required_engines"]:
            sample = by_engine[engine]
            accepted = ratio(summed(sample, "accepted"), summed(sample, "proposed"))
            calls = ratio(
                sum(len(row["output_ids"]) for row in sample), summed(sample, "target_calls")
            )
            bypass = ratio(
                summed(sample, "bypass_decisions"), summed(sample, "controller_decisions")
            )
            request_bypass = (
                None
                if any(r["bypass_decisions"] is None for r in sample)
                else sum(r["bypass_decisions"] > 0 for r in sample) / len(sample)
            )
            peaks = []
            for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
                values = [row[field] for row in sample]
                peaks.append(
                    None if any(value is None for value in values) else max(values) / 2**20
                )
            matched = None
            if group[2] == "greedy":
                reference_by_cell = {key(row)[:-1]: row["output_ids"] for row in reference}
                comparable = [row for row in sample if key(row)[:-1] in reference_by_cell]
                if comparable:
                    matched = sum(
                        row["output_ids"] == reference_by_cell[key(row)[:-1]] for row in comparable
                    ) / len(comparable)
            lines.append(
                "| "
                + " | ".join(
                    [
                        safe(engine),
                        "N/A" if matched is None else f"{100 * matched:.1f}%",
                        fmt(accepted),
                        fmt(calls),
                        fmt(bypass),
                        fmt(request_bypass),
                        fmt(peaks[0]),
                        fmt(peaks[1]),
                    ]
                )
                + " |"
            )
        lines += [
            "",
            "Speedups use per-prompt median latency, then a geometric mean. "
            "Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. "
            "p95 is descriptive, especially for small strata. N/A means unavailable or undefined.",
            "",
        ]
        if group[2] == "sample":
            lines += [
                "Sampled completions can have different lengths. End-to-end speedup "
                "alone is not a pure decode-rate comparison.",
                "",
            ]
        if "adaptive" in by_engine:
            _, low, high, _ = paired_speedup(reference, by_engine["adaptive"])
            conclusion = (
                "Adaptive latency improvement is supported by this interval."
                if low > 1
                else "Adaptive latency is worse than baseline across this interval."
                if high < 1
                else "Adaptive latency improvement is inconclusive at this interval."
            )
            lines += [conclusion, ""]
    if manifest.get("max_chosen_token_gap") is not None:
        lines += [
            "## Greedy conformance",
            "",
            "Greedy speculation is exact in real arithmetic. In BF16 it is not: a target-only "
            "step computes its logits in a width-1 forward while a verification step computes "
            "them inside a wider one, and when the top two logits are within that noise the "
            "argmax flips. The `Greedy match` column above is the measured fraction of requests "
            "whose token ids were identical to the baseline's.",
            "",
            f"Every mismatch in this run was checked against a near-tie bound of "
            f"{manifest['max_chosen_token_gap']} logits between the two tokens the engines "
            f"actually chose. A mismatch with no recorded evidence, or one beyond that bound, "
            f"is refused rather than reported. See docs/decisions/0004-bf16-greedy-conformance.md.",
            "",
        ]
    lines += [
        "## Limits",
        "",
        "These are sequential, batch-one warm-request results on the recorded configuration. "
        "They do not establish multi-user serving throughput, capability accuracy, "
        "or performance on other GPUs. "
        "Fixed-length conditions suppress EOS and must be read separately from natural stopping. "
        "Hashes check file integrity; they cannot prove honest acquisition. "
        "Consult SPEC.md and the acquisition code for measurement boundaries.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    root = args.run_directory.resolve()
    output = args.out.resolve()
    try:
        require(
            output
            not in {root / name for name in ("manifest.json", "requests.jsonl", "evidence.json")},
            "Output cannot overwrite raw evidence",
        )
        manifest, evidence, rows = load_run(root, args.allow_fixture)
        require(
            not (manifest["data_kind"] == "fixture" and output.name.lower() == "results.md"),
            "Fixture reports must never be named RESULTS.md",
        )
        report = render(manifest, evidence, rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent, prefix=".report-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(report)
            os.replace(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"Report refused: {error}\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
