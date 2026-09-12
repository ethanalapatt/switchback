"""Source and file provenance helpers.

Hashing establishes integrity of a recorded file, never the honesty of the
measurement that produced it (SPEC.md section 6). Nothing here interprets a
benchmark number.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Files whose content defines "the code that produced a run". Kept explicit so a
# run manifest cannot silently omit a source file that changed behaviour.
SOURCE_GLOBS: tuple[str, ...] = (
    "src/switchback/**/*.py",
    "bench/**/*.py",
    "scripts/*.py",
)


def repo_root() -> Path:
    """Directory containing ``pyproject.toml`` for the installed source tree."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml above {here}")


def sha256_file(path: Path) -> str:
    """SHA-256 of one file's bytes, as lowercase hex."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """SHA-256 of an in-memory payload, as lowercase hex."""
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@dataclass(frozen=True)
class SourceProvenance:
    """Identity of the code that produced an artifact.

    ``commit`` is ``None`` when the tree has no commits yet; the report renderer
    refuses such a run rather than inventing a hash.
    """

    commit: str | None
    dirty: bool | None
    source_sha256: str
    file_count: int


def source_digest(root: Path) -> tuple[str, int]:
    """Hash every tracked source file's path and content.

    Returns ``(hex_digest, file_count)``. The recipe is: sort relative POSIX
    paths, then feed ``path + "\\0" + sha256(content) + "\\n"`` into SHA-256. It
    is reproducible from the repository alone and is documented in the run
    manifest so a reviewer can recompute it.
    """
    paths: list[Path] = []
    for pattern in SOURCE_GLOBS:
        paths.extend(p for p in root.glob(pattern) if p.is_file())
    unique = sorted({p.relative_to(root).as_posix() for p in paths})
    digest = hashlib.sha256()
    for relative in unique:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(root / relative).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(unique)


def collect_source_provenance(root: Path | None = None) -> SourceProvenance:
    """Resolve git identity plus a git-independent source digest."""
    base = root if root is not None else repo_root()
    commit = _git(base, "rev-parse", "HEAD")
    if commit is not None and len(commit) != 40:
        commit = None
    status = _git(base, "status", "--porcelain")
    dirty = None if status is None else bool(status.strip())
    digest, count = source_digest(base)
    return SourceProvenance(
        commit=commit, dirty=dirty, source_sha256=digest, file_count=count
    )
