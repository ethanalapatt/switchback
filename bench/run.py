"""Execute a benchmark matrix and write raw, resumable evidence.

SPEC.md sections 9.3 and 9.4. What this file is responsible for getting right:

* **Fairness.** Every engine sees the same prompt token ids, the same decode
  config, both models resident, and the same correctness-check setting. Engine
  order is randomized inside each prompt/repeat block from a recorded seed.
* **Timing boundaries.** The host timer starts after a GPU synchronization and
  before prefill, and stops only after GPU completion. Serialization happens
  after the timed region.
* **Nothing is dropped.** A request that errors is written with its status and
  error text. The renderer refuses such a cohort rather than letting it vanish.
* **Resumability.** Completed request keys are read back from the existing
  ``requests.jsonl`` and skipped, so an interrupted run continues instead of
  starting over. Resuming is only legitimate against the same frozen source; the
  manifest records the code hash and the renderer checks it.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.prepare import load_config, write_json
from switchback.calibration import load_profile
from switchback.controller import CostController
from switchback.decoder import (
    EngineOptions,
    decode_adaptive_greedy,
    decode_speculative_greedy,
    decode_speculative_sampled,
    decode_target_only,
)
from switchback.env import collect_environment
from switchback.models import hf_baselines
from switchback.models.qwen import QwenAdapter
from switchback.provenance import collect_source_provenance, sha256_file
from switchback.types import DecodeConfig, DecodeResult

KEY_FIELDS = (
    "cohort",
    "dataset",
    "mode",
    "condition",
    "prompt_id",
    "seed",
    "repeat",
    "engine",
)


@dataclass(frozen=True)
class RequestKey:
    """The identity of one logical request, matching the renderer's key."""

    cohort: str
    dataset: str
    mode: str
    condition: str
    prompt_id: str
    seed: int
    repeat: int
    engine: str

    def as_list(self) -> list[Any]:
        return [
            self.cohort,
            self.dataset,
            self.mode,
            self.condition,
            self.prompt_id,
            self.seed,
            self.repeat,
            self.engine,
        ]


def _memory(device: str) -> tuple[int | None, int | None]:
    import torch

    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return None, None
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def _reset_memory(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def build_engines(
    target: Any,
    draft: Any,
    target_adapter: QwenAdapter,
    draft_adapter: QwenAdapter,
    profile: Any,
    config: DecodeConfig,
    device: str,
    options: EngineOptions,
) -> dict[str, Any]:
    """Every engine in SPEC.md section 9.2, behind one uniform call shape."""
    eos = target.eos_token_ids
    pad = target.tokenizer.pad_token_id
    sampled = config.mode == "sample"

    def hf_ar(ids: list[int]) -> Any:
        return hf_baselines.run_hf_ar(target.model, ids, config, eos, pad, device)

    def hf_dynamic(ids: list[int]) -> Any:
        return hf_baselines.run_hf_dynamic(target.model, draft.model, ids, config, eos, pad, device)

    def native_ar(ids: list[int]) -> DecodeResult:
        return decode_target_only(target_adapter, ids, config, eos, options=options)

    def fixed(gamma: int) -> Any:
        def run(ids: list[int]) -> DecodeResult:
            engine = decode_speculative_sampled if sampled else decode_speculative_greedy
            return engine(
                target_adapter,
                draft_adapter,
                ids,
                config,
                eos,
                gamma=gamma,
                options=options,
            )

        return run

    def adaptive(ids: list[int]) -> DecodeResult:
        controller = CostController(profile=profile, allow_bypass=True)
        return decode_adaptive_greedy(
            target_adapter, draft_adapter, ids, config, eos, controller, options=options
        )

    def adaptive_no_bypass(ids: list[int]) -> DecodeResult:
        controller = CostController(profile=profile, allow_bypass=False)
        return decode_adaptive_greedy(
            target_adapter, draft_adapter, ids, config, eos, controller, options=options
        )

    return {
        "hf_ar": hf_ar,
        "native_ar": native_ar,
        "fixed_1": fixed(1),
        "fixed_2": fixed(2),
        "fixed_4": fixed(4),
        "fixed_8": fixed(8),
        "hf_dynamic": hf_dynamic,
        "adaptive": adaptive,
        "adaptive_no_bypass": adaptive_no_bypass,
    }


def row_from_result(key: RequestKey, result: Any, device: str) -> dict[str, Any]:
    """Turn an engine result into a raw request row, nulls and all."""
    allocated, reserved = _memory(device)
    counters = dict(getattr(result, "counters", {}) or {})
    return {
        **dict(zip(KEY_FIELDS, key.as_list(), strict=True)),
        "status": "ok",
        "start_ns": int(result.start_ns),
        "first_token_ns": int(result.first_token_ns),
        "last_token_ns": int(result.last_token_ns),
        "end_ns": int(result.end_ns),
        "output_ids": [int(value) for value in result.output_ids],
        "accepted": counters.get("accepted", getattr(result, "accepted", None)),
        "proposed": counters.get("proposed", getattr(result, "proposed", None)),
        "target_calls": counters.get("target_calls", getattr(result, "target_calls", None)),
        "bypass_decisions": counters.get(
            "bypass_decisions", getattr(result, "bypass_decisions", None)
        ),
        "controller_decisions": counters.get(
            "controller_decisions", getattr(result, "controller_decisions", None)
        ),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "draft_calls": getattr(result, "draft_calls", None),
        "prompt_tokens": counters.get("prompt_tokens"),
    }


def error_row(key: RequestKey, error: BaseException) -> dict[str, Any]:
    """A failed request is recorded, never dropped (SPEC.md section 9.4)."""
    return {
        **dict(zip(KEY_FIELDS, key.as_list(), strict=True)),
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "start_ns": 0,
        "first_token_ns": 0,
        "last_token_ns": 0,
        "end_ns": 0,
        "output_ids": [],
        "accepted": None,
        "proposed": None,
        "target_calls": None,
        "bypass_decisions": None,
        "controller_decisions": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }


def attach_divergences(
    rows: Sequence[dict[str, Any]],
    baseline: str,
    target_adapter: QwenAdapter,
    prompt_ids: Sequence[int],
    forbidden: Sequence[int],
    device: str,
) -> None:
    """Record why a greedy engine disagreed with the baseline, if it did.

    Computed after the timed region, from an uncached recomputation with the
    same EOS mask the engines used. ADR 0005 requires this evidence before the
    renderer will admit a mismatch.
    """
    from switchback.conformance import _divergence_detail

    reference = next((row for row in rows if row["engine"] == baseline), None)
    if reference is None or reference["status"] != "ok":
        return
    expected = reference["output_ids"]
    for row in rows:
        if row["status"] != "ok" or row["output_ids"] == expected:
            continue
        index = next(
            (
                position
                for position, (left, right) in enumerate(
                    zip(expected, row["output_ids"], strict=False)
                )
                if left != right
            ),
            min(len(expected), len(row["output_ids"])),
        )
        if index >= len(expected) or index >= len(row["output_ids"]):
            # One output is a strict prefix of the other. That is a length
            # disagreement, not a near-tie, and it gets no evidence attached so
            # the renderer refuses it.
            continue
        detail = _divergence_detail(
            target_adapter,
            [*prompt_ids, *expected[:index]],
            forbidden,
            int(expected[index]),
            int(row["output_ids"][index]),
            device,
        )
        row["greedy_divergence"] = {
            "index": index,
            "reference_token": int(expected[index]),
            "candidate_token": int(row["output_ids"][index]),
            "chosen_token_gap": abs(detail.reference_logit - detail.candidate_logit),
            "logit_scale": detail.logit_scale,
            "top2_margin": detail.margin,
        }


def load_completed(path: Path) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    """Read back an interrupted run so it can continue where it stopped."""
    if not path.exists():
        return [], set()
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return rows, {tuple(row[name] for name in KEY_FIELDS) for row in rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--workload", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=Path("artifacts/calibration.json"))
    parser.add_argument("--evidence", type=Path, default=Path("artifacts/evidence.json"))
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="stop cleanly after this long; rerun to resume",
    )
    args = parser.parse_args(argv)

    from switchback.manifest import load_manifest
    from switchback.pilot import load_pair
    from switchback.runtime import configure_torch_native_overrides, deterministic_runtime
    from switchback.sampling import forbidden_token_ids

    config, config_sha = load_config(args.config)
    workload_path = args.workload or Path(f"artifacts/workloads/{config['name']}.json")
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    tokens = json.loads(
        workload_path.with_name(f"{config['name']}.tokens.json").read_text(encoding="utf-8")
    )["prompt_token_ids"]

    native_status = configure_torch_native_overrides().as_dict()
    knobs = deterministic_runtime()
    environment = collect_environment(require_gpu=config["models"]["device"].startswith("cuda"))
    if not environment.ok:
        print("doctor checks failed; refusing to benchmark", file=sys.stderr)
        return 1

    manifest_document = load_manifest(args.manifest)
    target_entry = manifest_document["models"]["target"]
    draft_entry = manifest_document["models"]["draft"]
    target, draft, _, load_timings = load_pair(
        target_entry["repo_id"],
        target_entry["revision"],
        draft_entry["repo_id"],
        draft_entry["revision"],
        device=config["models"]["device"],
        dtype=config["models"]["dtype"],
        attn_implementation=config["models"]["attn_implementation"],
        local_files_only=True,
    )
    device = config["models"]["device"]
    target_adapter = QwenAdapter(
        model=target.model, device=device, vocab_size=target.logits_vocab_size, name="target"
    )
    draft_adapter = QwenAdapter(
        model=draft.model, device=device, vocab_size=draft.logits_vocab_size, name="draft"
    )
    profile = load_profile(args.calibration)

    condition = config["condition"]
    decode_config = DecodeConfig(
        mode=condition["mode"],
        temperature=float(condition["temperature"]),
        max_new_tokens=int(condition["max_new_tokens"]),
        eos_policy=condition["eos_policy"],
        seed=int(condition["seeds"][0]),
    )
    options = EngineOptions(correctness_checks=True, device=device)
    engines = build_engines(
        target, draft, target_adapter, draft_adapter, profile, decode_config, device, options
    )
    required = list(config["run"]["engines"])
    missing = [name for name in required if name not in engines]
    if missing:
        print(f"unknown engines in config: {missing}", file=sys.stderr)
        return 2

    heldout = [prompt for prompt in workload["prompts"] if prompt["role"] == "heldout"]
    seeds = [int(value) for value in condition["seeds"]]
    repeats = int(condition["repeats"])
    expected: list[RequestKey] = [
        RequestKey(
            cohort=config["cohort"],
            dataset=prompt["dataset"],
            mode=decode_config.mode,
            condition=condition["name"],
            prompt_id=prompt["prompt_id"],
            seed=seed,
            repeat=repeat,
            engine=engine,
        )
        for prompt in heldout
        for seed in seeds
        for repeat in range(repeats)
        for engine in required
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    requests_path = args.out / "requests.jsonl"
    rows, done = load_completed(requests_path)
    forbidden = forbidden_token_ids(decode_config, target.eos_token_ids)

    # Resident floor, before any request allocates a cache.
    _reset_memory(device)
    resident_allocated, resident_reserved = _memory(device)

    # Warmups, untimed and unrecorded.
    warm_prompt = tokens[heldout[0]["prompt_id"]]
    for _ in range(int(config["run"]["warmups"])):
        for name in required:
            engines[name](list(warm_prompt))

    rng = random.Random(int(config["run"]["scheduling_seed"]))
    started = time.perf_counter()
    stopped_early = False
    baseline = config["run"]["baseline"]

    with requests_path.open("a", encoding="utf-8") as stream:
        for seed in seeds:
            for repeat in range(repeats):
                for prompt in heldout:
                    order = list(required)
                    rng.shuffle(order)
                    block: list[dict[str, Any]] = []
                    for engine in order:
                        key = RequestKey(
                            cohort=config["cohort"],
                            dataset=prompt["dataset"],
                            mode=decode_config.mode,
                            condition=condition["name"],
                            prompt_id=prompt["prompt_id"],
                            seed=seed,
                            repeat=repeat,
                            engine=engine,
                        )
                        if tuple(key.as_list()) in done:
                            continue
                        _reset_memory(device)
                        try:
                            result = engines[engine](list(tokens[prompt["prompt_id"]]))
                            block.append(row_from_result(key, result, device))
                        except Exception as error:
                            block.append(error_row(key, error))
                    if block and decode_config.mode == "greedy":
                        attach_divergences(
                            block,
                            baseline,
                            target_adapter,
                            tokens[prompt["prompt_id"]],
                            forbidden,
                            device,
                        )
                    for row in block:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        rows.append(row)
                        done.add(tuple(row[name] for name in KEY_FIELDS))
                    stream.flush()
                    if args.max_seconds and time.perf_counter() - started > args.max_seconds:
                        stopped_early = True
                        break
                if stopped_early:
                    break
            if stopped_early:
                break

    complete = {tuple(key.as_list()) for key in expected} <= done
    provenance = collect_source_provenance()
    run_manifest = {
        "schema_version": 1,
        "data_kind": "measured",
        "complete": bool(complete and not stopped_early),
        "git_commit": provenance.commit,
        "git_dirty": provenance.dirty,
        "source_sha256": provenance.source_sha256,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cohort": config["cohort"],
        "condition": condition,
        "baseline": baseline,
        "required_engines": required,
        "max_chosen_token_gap": config["run"].get("max_chosen_token_gap"),
        "expected_keys": [key.as_list() for key in expected],
        "hardware": {
            "hostname": platform.node(),
            "device": environment.torch.get("device_name"),
            "capability": environment.torch.get("device_capability"),
            "total_memory_bytes": environment.torch.get("total_memory_bytes"),
            "driver_version": environment.driver.get("driver_version"),
            "machine": environment.host.get("machine"),
            "resident_allocated_bytes": resident_allocated,
            "resident_reserved_bytes": resident_reserved,
        },
        "software": {
            **environment.packages,
            "python": environment.python["version"],
            "cuda_build": environment.torch.get("cuda_build_version"),
            "attn_implementation": config["models"]["attn_implementation"],
            "dtype": config["models"]["dtype"],
            "torch_native_overrides": native_status,
            "deterministic_knobs": knobs,
        },
        "model_revisions": {
            "target": f"{target_entry['repo_id']}@{target_entry['revision']}",
            "draft": f"{draft_entry['repo_id']}@{draft_entry['revision']}",
        },
        "setup_seconds": load_timings,
        "config_sha256": config_sha,
        "workload_sha256": workload["workload_sha256"],
        "calibration_sha256": profile.sha256(),
        "benchmark_source_sha256": provenance.source_sha256,
        "scheduling_seed": int(config["run"]["scheduling_seed"]),
    }

    evidence_source = args.evidence
    if evidence_source.exists():
        (args.out / "evidence.json").write_text(
            evidence_source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    else:
        print(f"no evidence at {evidence_source}; run `switchback evidence --gpu`", file=sys.stderr)
        return 3

    run_manifest["files"] = {
        name: sha256_file(args.out / name) for name in ("requests.jsonl", "evidence.json")
    }
    write_json(args.out / "manifest.json", run_manifest)

    errors = sum(1 for row in rows if row.get("status") != "ok")
    print(f"  cohort      {config['cohort']} / {condition['name']}")
    print(f"  engines     {len(required)}   prompts {len(heldout)}   repeats {repeats}")
    print(f"  requests    {len(rows)} of {len(expected)} expected")
    print(f"  errors      {errors}")
    print(f"  complete    {run_manifest['complete']}")
    if stopped_early:
        print("  stopped early on --max-seconds; rerun with the same arguments to resume")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
