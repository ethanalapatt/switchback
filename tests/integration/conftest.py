"""Shared fixtures for tests that need the real pinned checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from switchback.manifest import load_manifest
from switchback.provenance import repo_root

MANIFEST_PATH = repo_root() / "data" / "manifest.json"


@pytest.fixture(scope="session")
def manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        pytest.skip(f"no {MANIFEST_PATH}; run `python -m switchback resolve-models`")
    return load_manifest(Path(MANIFEST_PATH))


@pytest.fixture(scope="session")
def loaded_pair(manifest: dict[str, Any]) -> Any:
    """Target and draft resident on the GPU, loaded once for the whole session."""
    from switchback.pilot import load_pair
    from switchback.runtime import configure_torch_native_overrides, deterministic_runtime

    configure_torch_native_overrides()
    deterministic_runtime()
    target, draft, compatibility, _ = load_pair(
        manifest["models"]["target"]["repo_id"],
        manifest["models"]["target"]["revision"],
        manifest["models"]["draft"]["repo_id"],
        manifest["models"]["draft"]["revision"],
        device="cuda",
        dtype="bfloat16",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    return target, draft, compatibility
