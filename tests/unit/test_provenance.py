"""The source digest must change when behaviour-defining code changes."""

from __future__ import annotations

from pathlib import Path

from switchback.provenance import (
    collect_source_provenance,
    repo_root,
    sha256_bytes,
    sha256_file,
    source_digest,
)


def build_tree(root: Path, body: str) -> None:
    package = root / "src" / "switchback"
    package.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='fake'\n")
    (package / "decoder.py").write_text(body)


def test_digest_is_stable_for_identical_trees(tmp_path: Path) -> None:
    left, right = tmp_path / "a", tmp_path / "b"
    build_tree(left, "x = 1\n")
    build_tree(right, "x = 1\n")
    assert source_digest(left) == source_digest(right)


def test_digest_changes_when_content_changes(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    build_tree(root, "x = 1\n")
    before, count = source_digest(root)
    build_tree(root, "x = 2\n")
    after, recount = source_digest(root)
    assert before != after
    assert count == recount == 1


def test_digest_changes_when_a_file_is_renamed(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    build_tree(root, "x = 1\n")
    before, _ = source_digest(root)
    (root / "src" / "switchback" / "decoder.py").rename(root / "src" / "switchback" / "sampling.py")
    after, _ = source_digest(root)
    assert before != after


def test_digest_ignores_files_outside_the_source_globs(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    build_tree(root, "x = 1\n")
    before, _ = source_digest(root)
    (root / "README.md").write_text("documentation only")
    after, _ = source_digest(root)
    assert before == after


def test_file_and_byte_digests_agree(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    payload = b"switchback" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


def test_repo_root_contains_the_project_file() -> None:
    assert (repo_root() / "pyproject.toml").is_file()


def test_collected_provenance_describes_this_checkout() -> None:
    provenance = collect_source_provenance()
    assert provenance.file_count > 0
    assert len(provenance.source_sha256) == 64
    assert provenance.commit is None or len(provenance.commit) == 40
