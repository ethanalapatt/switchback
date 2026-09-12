"""Lock the workload: which prompts, from which dataset revisions, in what order.

SPEC.md section 9.1. Rows are selected by sorting the SHA-256 of
``source_revision + split + stable_row_id`` and taking the required count. No
output is ever inspected to choose a prompt, and the calibration and held-out
splits come from different dataset splits so nothing the controller was fitted
on can appear in the evaluation.

The workload file records prompt ids, token counts and a hash of the token ids.
It does **not** record the prompt text: a workload should be shareable without
redistributing dataset content, and the hash pins which prompt it was.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchback.provenance import collect_source_provenance, sha256_bytes

WORKLOAD_SCHEMA_VERSION = 1

CODE_INSTRUCTION = "Write a Python function for this task. Return code only."
MATH_INSTRUCTION = "Solve the problem and state the final answer."

# Immutable dataset revisions, resolved 2026-09-12. Both are public.
DATASETS: dict[str, dict[str, Any]] = {
    "mbpp": {
        "repo_id": "google-research-datasets/mbpp",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "config": "full",
        "license": "cc-by-4.0",
        "calibration_split": "train",
        "heldout_split": "test",
        "text_column": "text",
        "id_column": "task_id",
        "instruction": CODE_INSTRUCTION,
    },
    "gsm8k": {
        "repo_id": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "config": "main",
        "license": "mit",
        "calibration_split": "train",
        "heldout_split": "test",
        "text_column": "question",
        # GSM8K has no stable row id, so source order at the locked revision is
        # the identity (SPEC.md section 9.1 permits this explicitly).
        "id_column": None,
        "instruction": MATH_INSTRUCTION,
    },
}


@dataclass(frozen=True)
class WorkloadPrompt:
    """One locked prompt. ``token_sha256`` pins it without storing its text."""

    prompt_id: str
    dataset: str
    split: str
    role: str
    source_row_id: str
    prompt_tokens: int
    token_sha256: str
    target_tokens: int | None = None


def _read_split(entry: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Download one split at the locked revision and read it."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - optional extra
        raise SystemExit(
            "bench.prepare needs pyarrow: python -m pip install -e '.[bench]'"
        ) from error
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=entry["repo_id"],
        filename=f"{entry['config']}/{split}-00000-of-00001.parquet",
        revision=entry["revision"],
        repo_type="dataset",
    )
    return pq.read_table(path).to_pylist()


def select_rows(
    rows: Sequence[dict[str, Any]],
    revision: str,
    split: str,
    id_column: str | None,
    count: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Pick ``count`` rows by hash order, never by looking at their content.

    Returns ``(stable_row_id, row)`` pairs. The ordering key is the SHA-256 of
    ``revision + split + row_id``, so the selection is reproducible from the
    manifest alone and does not depend on this code's iteration order.
    """
    if count > len(rows):
        raise SystemExit(f"asked for {count} rows from {split} but it has only {len(rows)}")
    keyed: list[tuple[str, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        row_id = str(row[id_column]) if id_column else str(index)
        digest = hashlib.sha256(f"{revision}{split}{row_id}".encode()).hexdigest()
        keyed.append((digest, row_id, row))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [(row_id, row) for _, row_id, row in keyed[:count]]


def build_prompts(
    tokenizer: Any,
    config: dict[str, Any],
) -> tuple[list[WorkloadPrompt], dict[str, list[int]], dict[str, Any]]:
    """Assemble every prompt the config asks for, in a deterministic order."""
    from data.synthetic_generator import generate

    from switchback.models.qwen import render_chat_prompt

    workload = config["workload"]
    prompts: list[WorkloadPrompt] = []
    token_ids: dict[str, list[int]] = {}
    dataset_records: dict[str, Any] = {}

    for dataset, counts in (
        ("mbpp", ("code_calibration", "code_heldout")),
        ("gsm8k", ("math_calibration", "math_heldout")),
    ):
        entry = DATASETS[dataset]
        dataset_records[dataset] = {
            name: entry[name] for name in ("repo_id", "revision", "config", "license")
        }
        for role, key in (("calibration", counts[0]), ("heldout", counts[1])):
            wanted = int(workload[key])
            if wanted == 0:
                continue
            split = entry[f"{role}_split"]
            rows = _read_split(entry, split)
            for row_id, row in select_rows(
                rows, entry["revision"], split, entry["id_column"], wanted
            ):
                text = f"{entry['instruction']}\n\n{row[entry['text_column']].strip()}"
                ids = render_chat_prompt(tokenizer, text)
                prompt_id = f"{dataset}_{role}_{row_id}"
                token_ids[prompt_id] = ids
                prompts.append(
                    WorkloadPrompt(
                        prompt_id=prompt_id,
                        dataset=dataset,
                        split=split,
                        role=role,
                        source_row_id=row_id,
                        prompt_tokens=len(ids),
                        token_sha256=sha256_bytes(json.dumps(ids).encode()),
                    )
                )

    for role, key, seed_key in (
        ("calibration", "controlled_calibration", "controlled_seed_calibration"),
        ("heldout", "controlled_heldout", "controlled_seed_heldout"),
    ):
        wanted = int(workload.get(key, 0))
        if wanted == 0:
            continue
        for generated in generate(seed=int(workload[seed_key]), count=wanted):
            ids = render_chat_prompt(tokenizer, generated.text)
            prompt_id = f"controlled_{role}_{generated.prompt_id}"
            token_ids[prompt_id] = ids
            prompts.append(
                WorkloadPrompt(
                    prompt_id=prompt_id,
                    dataset="controlled",
                    split=role,
                    role=role,
                    source_row_id=generated.prompt_id,
                    prompt_tokens=len(ids),
                    token_sha256=sha256_bytes(json.dumps(ids).encode()),
                    target_tokens=generated.target_tokens,
                )
            )
        dataset_records["controlled"] = {
            "repo_id": "switchback/controlled",
            "revision": f"generator-seed-{workload[seed_key]}",
            "config": "deterministic",
            "license": "Apache-2.0",
        }

    return prompts, token_ids, dataset_records


def workload_document(
    config_name: str,
    config_sha256: str,
    prompts: Sequence[WorkloadPrompt],
    datasets: dict[str, Any],
) -> dict[str, Any]:
    """The workload file: identities and hashes, never prompt text."""
    provenance = collect_source_provenance()
    body = {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "config_name": config_name,
        "config_sha256": config_sha256,
        "datasets": datasets,
        "prompts": [asdict(prompt) for prompt in prompts],
        "selection_recipe": (
            "Sort rows by SHA-256 of (revision + split + stable_row_id), then "
            "take the required count. GSM8K has no stable row id, so source "
            "order at the locked revision is used."
        ),
    }
    body["workload_sha256"] = sha256_bytes(json.dumps(body, sort_keys=True).encode())
    body["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    body["source"] = {
        "commit": provenance.commit,
        "dirty": provenance.dirty,
        "source_sha256": provenance.source_sha256,
    }
    return body


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    """Read a config and hash its exact bytes."""
    raw = path.read_bytes()
    config = tomllib.loads(raw.decode("utf-8"))
    if config.get("schema_version") != 1:
        raise SystemExit(f"{path}: unsupported config schema_version")
    return config, sha256_bytes(raw)


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tokens-out", type=Path, default=None)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from switchback.manifest import load_manifest, write_manifest

    config, config_sha = load_config(args.config)
    manifest = load_manifest(args.manifest)
    target = manifest["models"]["target"]
    tokenizer = AutoTokenizer.from_pretrained(target["repo_id"], revision=target["revision"])
    prompts, token_ids, datasets = build_prompts(tokenizer, config)
    document = workload_document(config["name"], config_sha, prompts, datasets)

    out = args.out or Path(f"artifacts/workloads/{config['name']}.json")
    tokens_out = args.tokens_out or out.with_name(f"{config['name']}.tokens.json")
    write_json(out, document)
    write_json(tokens_out, {"prompt_token_ids": token_ids})

    manifest["datasets"] = datasets
    write_manifest(manifest, args.manifest)

    by_role: dict[str, int] = {}
    for prompt in prompts:
        by_role[prompt.role] = by_role.get(prompt.role, 0) + 1
    print(f"  config           {config['name']} (sha256 {config_sha[:16]}...)")
    print(f"  workload sha256  {document['workload_sha256']}")
    for role, count in sorted(by_role.items()):
        print(f"  {role:<15}  {count} prompts")
    lengths = [prompt.prompt_tokens for prompt in prompts]
    if lengths:
        print(
            f"  prompt tokens    min {min(lengths)}  median "
            f"{sorted(lengths)[len(lengths) // 2]}  max {max(lengths)}"
        )
    print(f"  wrote {out}")
    print(f"  wrote {tokens_out}")
    print(f"  updated {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
