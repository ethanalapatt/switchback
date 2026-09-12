"""The manifest must refuse anything that is not an immutable pin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from switchback.manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    load_manifest,
    write_manifest,
)
from switchback.types import ConfigError


def entry(revision: str) -> dict[str, object]:
    return {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": revision,
        "license": "apache-2.0",
        "metadata_sha256": {"config.json": "0" * 64},
        "weight_files": ["model.safetensors"],
        "weight_bytes": 10,
    }


def test_round_trip(tmp_path: Path) -> None:
    document = build_manifest(entry("a" * 40), entry("b" * 40))
    path = tmp_path / "manifest.json"
    write_manifest(document, path)
    loaded = load_manifest(path)
    assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert loaded["models"]["target"]["revision"] == "a" * 40
    assert loaded["datasets"] == {}


def test_written_manifest_is_sorted_and_newline_terminated(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(build_manifest(entry("a" * 40), entry("b" * 40)), path)
    text = path.read_text()
    assert text.endswith("\n")
    assert json.loads(text) == json.loads(text)
    assert text.index('"datasets"') < text.index('"models"')


def test_branch_name_instead_of_a_commit_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(build_manifest(entry("main"), entry("b" * 40)), path)
    with pytest.raises(ConfigError, match="40-character revision"):
        load_manifest(path)


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    document = build_manifest(entry("a" * 40), entry("b" * 40))
    document["schema_version"] = 99
    path.write_text(json.dumps(document))
    with pytest.raises(ConfigError, match="schema_version"):
        load_manifest(path)


def test_missing_draft_entry_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    document = build_manifest(entry("a" * 40), entry("b" * 40))
    del document["models"]["draft"]
    path.write_text(json.dumps(document))
    with pytest.raises(ConfigError, match=r"models\.target and models\.draft"):
        load_manifest(path)
