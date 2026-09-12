"""Adjudicate greedy divergences the cheap screen could not explain.

`bench.run` records, for every greedy mismatch, the logit gap between the two
tokens the engines chose, measured by recomputing the row **uncached** from the
baseline's prefix. That screen is cheap and it runs inside the benchmark.

The primary run falsified it. Two positions came back with gaps of 1.5 and 0.75
logits, apparently far outside BF16 reduction-order noise. Investigating them
showed the screen was measuring the wrong thing:

* At ``mbpp_heldout_169`` index 46 the uncached row prefers the baseline's token
  by 1.5, and every freshly-prefilled cached row prefers it by 0.5 -- but in the
  speculative engine's **own** incrementally-built cache the two tokens were
  43.75 and 44.00, a margin of 0.25 the other way. The engine chose correctly
  given its own logits. Its KV cache was built through 46 blocks of varying
  widths and crops, so the cached keys and values themselves differ in their
  last bits from a single-shot prefill.
* At ``gsm8k_heldout_251`` index 72 **every** reproducible path prefers the
  token ``hf_dynamic`` chose. The baseline is the outlier there, not the
  candidate.

The lesson is that "the gap" is not a property of a position. It is a property
of a position *and an execution path*, and the baseline is one path among
several rather than ground truth.

So this module recomputes each flagged position under every path it can
reproduce -- uncached, freshly-cached at several verification widths, a cache
grown in chunks, and for Switchback engines a full deterministic replay of the
engine itself -- and reports the smallest margin any of them produced.

A divergence is adjudicated ``near_tie`` when some reproducible path puts the
two tokens within the bound, or when the paths do not unanimously agree with the
baseline. Either condition means the model was near-indifferent and the engines
were choosing between near-equals. A genuine cache fault looks nothing like
this: the candidate token sits far below the reference under every path and no
path prefers it, and that is adjudicated ``unexplained``.

Run after a benchmark. It reads the timings but never changes them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.prepare import load_config
from bench.run import KEY_FIELDS
from switchback.models.qwen import QwenAdapter
from switchback.provenance import collect_source_provenance, sha256_file

ADJUDICATION_SCHEMA_VERSION = 1

# Verification widths a speculative engine actually uses, plus width 1 for the
# target-only path.
PATH_WIDTHS: tuple[int, ...] = (1, 2, 3, 5, 9)
CHUNK = 5


@dataclass(frozen=True)
class PathMargin:
    """What one reproducible execution path thought of the two tokens."""

    path: str
    reference_logit: float
    candidate_logit: float

    @property
    def margin(self) -> float:
        return self.reference_logit - self.candidate_logit

    @property
    def prefers_candidate(self) -> bool:
        return self.candidate_logit > self.reference_logit


@dataclass(frozen=True)
class Adjudication:
    """The verdict on one flagged divergence."""

    cohort: str
    dataset: str
    mode: str
    condition: str
    prompt_id: str
    seed: int
    repeat: int
    engine: str
    index: int
    reference_token: int
    candidate_token: int
    screen_gap: float
    smallest_margin: float
    any_path_prefers_candidate: bool
    engine_replay_margin: float | None
    verdict: str
    paths: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_logits(adapter: QwenAdapter, prefix: Sequence[int], device: str) -> dict[str, Any]:
    """Compute the next-token row at ``prefix`` under every reproducible path."""
    import torch

    adapter.checks = False
    body = list(prefix)
    out: dict[str, Any] = {}
    with torch.inference_mode():
        out["uncached"] = (
            adapter.model(
                input_ids=torch.tensor([body], dtype=torch.long, device=device),
                use_cache=False,
            )
            .logits[0, -1]
            .float()
        )
        for width in PATH_WIDTHS:
            cache = adapter.new_cache()
            adapter.forward(torch.tensor([body[:-1]], dtype=torch.long, device=device), cache)
            block = [body[-1], *([1] * (width - 1))]
            out[f"cached_width_{width}"] = adapter.forward(
                torch.tensor([block], dtype=torch.long, device=device), cache
            )[0, 0].float()
        cache = adapter.new_cache()
        for start in range(0, len(body) - 1, CHUNK):
            piece = body[start : min(start + CHUNK, len(body) - 1)]
            adapter.forward(torch.tensor([piece], dtype=torch.long, device=device), cache)
        out[f"cache_grown_in_{CHUNK}s"] = adapter.forward(
            torch.tensor([[body[-1]]], dtype=torch.long, device=device), cache
        )[0, -1].float()
    return out


def _replay_engine_margin(
    engine: str,
    target: QwenAdapter,
    draft: QwenAdapter,
    prompt_ids: Sequence[int],
    eos_token_ids: Sequence[int],
    index: int,
    reference_token: int,
    candidate_token: int,
    device: str,
) -> float | None:
    """Replay a Switchback engine and read the margin it actually decided on.

    This is the faithful measurement: it reproduces the engine's own cache,
    built through the same sequence of forwards and crops. Returns ``None`` for
    an external baseline, whose internals are not instrumentable from here.
    """
    import switchback.decoder as decoder
    from switchback.controller import CostController, FixedController
    from switchback.decoder import EngineOptions, decode_greedy_with_controller
    from switchback.types import DecodeConfig

    if engine.startswith("fixed_"):
        controller: Any = FixedController(int(engine.split("_")[1]))
    elif engine == "native_ar":
        controller = FixedController(0)
    elif engine.startswith("adaptive"):
        from switchback.calibration import load_profile

        controller = CostController(
            profile=load_profile(Path("artifacts/calibration.json")),
            allow_bypass=engine == "adaptive",
        )
        controller.reset()
    else:
        return None

    captured: list[tuple[int, float, float]] = []
    original = decoder.greedy_token

    def spy(row: Any, forbidden_ids: Sequence[int] = ()) -> int:
        chosen = original(row, forbidden_ids=forbidden_ids)
        values = row.float()
        captured.append((chosen, float(values[reference_token]), float(values[candidate_token])))
        return chosen

    decoder.greedy_token = spy
    try:
        result = decode_greedy_with_controller(
            target,
            draft,
            list(prompt_ids),
            DecodeConfig("greedy", 1.0, index + 1, "suppress_until_budget", 42),
            list(eos_token_ids),
            controller,
            options=EngineOptions(correctness_checks=True, device=device),
        )
    finally:
        decoder.greedy_token = original

    if len(result.output_ids) <= index or result.output_ids[index] != candidate_token:
        # The replay did not reproduce the divergence, so it cannot speak to it.
        return None
    committing = [
        (chosen, ref, cand) for chosen, ref, cand in captured if chosen == candidate_token
    ]
    if not committing:
        return None
    _, ref_logit, cand_logit = committing[-1]
    return ref_logit - cand_logit


def adjudicate(
    run: Path,
    config_path: Path,
    tokens: dict[str, list[int]],
    target: QwenAdapter,
    draft: QwenAdapter,
    eos_token_ids: Sequence[int],
    device: str,
) -> list[Adjudication]:
    """Adjudicate every divergence whose screen gap exceeded the bound."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    bound = manifest.get("max_chosen_token_gap")
    if bound is None:
        raise SystemExit("the run declares no max_chosen_token_gap; nothing to adjudicate")
    rows = [
        json.loads(line)
        for line in (run / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = manifest["baseline"]
    by_cell: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        by_cell.setdefault(tuple(row[name] for name in KEY_FIELDS[:-1]), {})[row["engine"]] = row

    flagged = [
        row
        for row in rows
        if (row.get("greedy_divergence") or {}).get("chosen_token_gap", 0) > bound
    ]
    print(f"  {len(flagged)} divergence(s) above the {bound} screen bound")

    verdicts: list[Adjudication] = []
    cache: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in flagged:
        divergence = row["greedy_divergence"]
        index = int(divergence["index"])
        reference_token = int(divergence["reference_token"])
        candidate_token = int(divergence["candidate_token"])
        cell = tuple(row[name] for name in KEY_FIELDS[:-1])
        reference_row = by_cell[cell][baseline]
        prefix = [
            *tokens[row["prompt_id"]],
            *reference_row["output_ids"][:index],
        ]
        signature = (row["prompt_id"], index, reference_token, candidate_token)
        if signature not in cache:
            rows_by_path = _row_logits(target, prefix, device)
            cache[signature] = {
                name: PathMargin(
                    path=name,
                    reference_logit=float(values[reference_token]),
                    candidate_logit=float(values[candidate_token]),
                )
                for name, values in rows_by_path.items()
            }
        paths = cache[signature]

        replay = _replay_engine_margin(
            row["engine"],
            target,
            draft,
            tokens[row["prompt_id"]],
            eos_token_ids,
            index,
            reference_token,
            candidate_token,
            device,
        )
        margins = [abs(margin.margin) for margin in paths.values()]
        if replay is not None:
            margins.append(abs(replay))
        smallest = min(margins)
        prefers = any(margin.prefers_candidate for margin in paths.values()) or (
            replay is not None and replay < 0
        )
        verdict = "near_tie" if (smallest <= bound or prefers) else "unexplained"
        verdicts.append(
            Adjudication(
                **{name: row[name] for name in KEY_FIELDS},
                index=index,
                reference_token=reference_token,
                candidate_token=candidate_token,
                screen_gap=float(divergence["chosen_token_gap"]),
                smallest_margin=smallest,
                any_path_prefers_candidate=prefers,
                engine_replay_margin=replay,
                verdict=verdict,
                paths=[asdict(margin) for margin in paths.values()],
            )
        )
    return verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    args = parser.parse_args(argv)

    from switchback.manifest import load_manifest
    from switchback.pilot import load_pair
    from switchback.runtime import configure_torch_native_overrides, deterministic_runtime

    configure_torch_native_overrides()
    deterministic_runtime()
    config, _ = load_config(args.config)
    tokens = json.loads(
        Path(f"artifacts/workloads/{config['name']}.tokens.json").read_text(encoding="utf-8")
    )["prompt_token_ids"]
    models = load_manifest(args.manifest)["models"]
    target, draft, _, _ = load_pair(
        models["target"]["repo_id"],
        models["target"]["revision"],
        models["draft"]["repo_id"],
        models["draft"]["revision"],
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

    verdicts = adjudicate(
        args.run,
        args.config,
        tokens,
        target_adapter,
        draft_adapter,
        target.eos_token_ids,
        device,
    )
    path = args.run / "adjudication.jsonl"
    path.write_text(
        "".join(json.dumps(item.as_dict(), sort_keys=True) + "\n" for item in verdicts),
        encoding="utf-8",
    )

    provenance = collect_source_provenance()
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["adjudication.jsonl"] = sha256_file(path)
    manifest["adjudication"] = {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": provenance.commit,
        "note": (
            "Added after the run. It recomputes logits at flagged positions and "
            "changes no timing, token or counter recorded during the benchmark."
        ),
        "adjudicated": len(verdicts),
        "unexplained": sum(1 for item in verdicts if item.verdict == "unexplained"),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)

    for item in verdicts:
        print(
            f"  {item.verdict:<11} {item.engine:<11} {item.prompt_id:<22} rep{item.repeat} "
            f"idx {item.index:>3}  screen {item.screen_gap:.3f}  "
            f"smallest {item.smallest_margin:.3f}"
            + (
                f"  engine replay {item.engine_replay_margin:+.3f}"
                if item.engine_replay_margin is not None
                else "  (external baseline)"
            )
        )
    unexplained = [item for item in verdicts if item.verdict == "unexplained"]
    print()
    print(f"  adjudicated {len(verdicts)}, unexplained {len(unexplained)}")
    print(f"  wrote {path}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
