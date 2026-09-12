"""The immutable source manifest.

Records exactly which public checkpoints (and later, which dataset revisions) a
run used. Revisions are resolved commit SHAs; nothing here is hand-written.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchback.provenance import sha256_file
from switchback.types import ConfigError

MANIFEST_SCHEMA_VERSION = 1

# Small metadata files worth hashing directly. The revision SHA already pins
# content; these hashes let a reviewer verify a local cache without the network.
METADATA_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)


def describe_model(repo_id: str, revision: str, license_id: str) -> dict[str, Any]:
    """Resolve one checkpoint's local snapshot and hash its metadata files."""
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
        )
    )
    metadata_hashes: dict[str, str] = {}
    for name in METADATA_FILES:
        candidate = root / name
        if candidate.is_file():
            metadata_hashes[name] = sha256_file(candidate)
    weights = sorted(path.name for path in root.glob("*.safetensors"))
    weight_bytes = sum((root / name).stat().st_size for name in weights)
    if not weights:
        raise ConfigError(f"{repo_id}@{revision} has no safetensors weights locally")
    return {
        # The local snapshot path is deliberately omitted: it is machine
        # specific and would make a committed manifest unportable.
        "repo_id": repo_id,
        "revision": revision,
        "license": license_id,
        "metadata_sha256": metadata_hashes,
        "weight_files": weights,
        "weight_bytes": weight_bytes,
    }


def build_manifest(target: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    """Assemble the manifest document. ``datasets`` is filled in by M7 tooling."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": {"target": target, "draft": draft},
        "datasets": {},
        "notes": (
            "Revisions are resolved immutable commit SHAs from the Hugging Face "
            "Hub. Dataset entries are written by bench.prepare in milestone 7."
        ),
    }


def write_manifest(document: dict[str, Any], out: Path) -> None:
    """Write the manifest atomically, sorted, with a trailing newline."""
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, out)


def load_manifest(path: Path) -> dict[str, Any]:
    """Read and structurally validate a manifest written by :func:`write_manifest`."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ConfigError(
            f"manifest schema_version {document.get('schema_version')!r} is not "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    models = document.get("models")
    if not isinstance(models, dict) or {"target", "draft"} - set(models):
        raise ConfigError("manifest must define models.target and models.draft")
    for role, entry in models.items():
        revision = entry.get("revision", "")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ConfigError(f"manifest models.{role} lacks a 40-character revision")
    return document
