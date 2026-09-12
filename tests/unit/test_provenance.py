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


def test_dirty_ignores_untracked_files_but_the_digest_does_not(tmp_path: Path) -> None:
    """A benchmark writes artifacts into its own repository while it runs."""
    import subprocess

    root = tmp_path / "repo"
    build_tree(root, "x = 1\n")
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "Test"],
        ["add", "-A"],
        ["commit", "-q", "-m", "initial"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    before = collect_source_provenance(root)
    assert before.dirty is False
    assert before.commit is not None and len(before.commit) == 40

    (root / "artifacts").mkdir()
    (root / "artifacts" / "requests.jsonl").write_text("{}\n")
    after_artifact = collect_source_provenance(root)
    assert after_artifact.dirty is False
    assert after_artifact.source_sha256 == before.source_sha256

    # An untracked source file is still caught, by the digest rather than by git.
    (root / "src" / "switchback" / "sampling.py").write_text("y = 2\n")
    after_source = collect_source_provenance(root)
    assert after_source.source_sha256 != before.source_sha256
    assert after_source.file_count == before.file_count + 1

    # A tracked modification does set the flag.
    (root / "src" / "switchback" / "decoder.py").write_text("x = 99\n")
    assert collect_source_provenance(root).dirty is True


def test_a_modified_tracked_artifact_does_not_mark_the_tree_dirty(tmp_path: Path) -> None:
    """A benchmark rewrites its own evidence file; that is not a code change."""
    import subprocess

    root = tmp_path / "repo"
    build_tree(root, "x = 1\n")
    (root / "artifacts").mkdir()
    (root / "artifacts" / "evidence.json").write_text("{}\n")
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "Test"],
        ["add", "-A"],
        ["commit", "-q", "-m", "initial"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    assert collect_source_provenance(root).dirty is False
    (root / "artifacts" / "evidence.json").write_text('{"passed": true}\n')
    assert collect_source_provenance(root).dirty is False
    # A behaviour-defining file still does.
    (root / "src" / "switchback" / "decoder.py").write_text("x = 2\n")
    assert collect_source_provenance(root).dirty is True
