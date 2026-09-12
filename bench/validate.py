"""Validate a run directory before anything is rendered from it.

Two layers, on purpose. The renderer's own ``load_run`` enforces the report
contract: integrity hashes, complete coverage, coherent timings, paired engines,
and the greedy near-tie rule from ADR 0005. This module adds the checks that
need the *inputs* the run claims to have used, which the renderer never sees:

* the config file still hashes to the ``config_sha256`` the run recorded;
* the workload file still hashes to ``workload_sha256``;
* the expected request keys are exactly what that config and that workload
  imply, so a run cannot quietly narrow its own cohort and then report
  ``complete: true``.

Exits non-zero on any failure. It reports every problem it finds rather than
stopping at the first, because a partially fixed run is worth knowing about.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from bench.prepare import load_config
from bench.run import KEY_FIELDS, RequestKey
from switchback.provenance import repo_root, sha256_bytes


def _renderer() -> Any:
    """Load the report renderer as a module so its contract is reused, not copied."""
    path = repo_root() / "scripts" / "render_results.py"
    spec = importlib.util.spec_from_file_location("report_renderer", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_keys_for(config: dict[str, Any], workload: dict[str, Any]) -> list[list[Any]]:
    """Rebuild the expected coverage from the config and the workload alone."""
    condition = config["condition"]
    heldout = [prompt for prompt in workload["prompts"] if prompt["role"] == "heldout"]
    return [
        RequestKey(
            cohort=config["cohort"],
            dataset=prompt["dataset"],
            mode=condition["mode"],
            condition=condition["name"],
            prompt_id=prompt["prompt_id"],
            seed=int(seed),
            repeat=repeat,
            engine=engine,
        ).as_list()
        for prompt in heldout
        for seed in condition["seeds"]
        for repeat in range(int(condition["repeats"]))
        for engine in config["run"]["engines"]
    ]


def validate(run: Path, config_path: Path, workload_path: Path | None) -> list[str]:
    """Return the list of problems. Empty means the run is publishable."""
    problems: list[str] = []
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config, config_sha = load_config(config_path)

    if manifest.get("config_sha256") != config_sha:
        problems.append(
            f"config hash mismatch: run recorded {manifest.get('config_sha256')}, "
            f"{config_path} now hashes to {config_sha}"
        )

    path = workload_path or Path(f"artifacts/workloads/{config['name']}.json")
    workload = json.loads(path.read_text(encoding="utf-8"))
    body = {
        name: value
        for name, value in workload.items()
        if name not in ("workload_sha256", "generated_at", "source")
    }
    recomputed = sha256_bytes(json.dumps(body, sort_keys=True).encode())
    if recomputed != workload["workload_sha256"]:
        problems.append(f"workload {path} does not hash to its own recorded workload_sha256")
    if manifest.get("workload_sha256") != workload["workload_sha256"]:
        problems.append(
            f"workload hash mismatch: run recorded {manifest.get('workload_sha256')}, "
            f"workload file records {workload['workload_sha256']}"
        )

    expected = {tuple(key) for key in expected_keys_for(config, workload)}
    recorded = {tuple(key) for key in manifest.get("expected_keys", [])}
    if expected != recorded:
        problems.append(
            f"expected coverage disagrees with the config and workload: "
            f"{len(expected - recorded)} missing, {len(recorded - expected)} extra"
        )

    rows = [
        json.loads(line)
        for line in (run / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = [row for row in rows if row.get("status") != "ok"]
    for row in errors:
        key = tuple(row.get(name) for name in KEY_FIELDS)
        problems.append(f"failed request {key}: {row.get('error')}")

    try:
        _renderer().load_run(run, allow_fixture=False)
    except (ValueError, KeyError, TypeError, OSError) as error:
        problems.append(f"report contract: {error}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, default=None)
    args = parser.parse_args(argv)

    problems = validate(args.run, args.config, args.workload)
    if problems:
        print(f"  {args.run}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"    {problem}")
        return 1
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    print(f"  {args.run}: valid")
    print(f"    cohort    {manifest['cohort']} / {manifest['condition']['name']}")
    print(f"    engines   {len(manifest['required_engines'])}")
    print(f"    requests  {len(manifest['expected_keys'])}")
    print(f"    commit    {manifest['git_commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
