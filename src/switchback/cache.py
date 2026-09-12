"""KV cache handles and the one pending-token convention.

SPEC.md section 4.3 fixes a single cache indexing contract for the whole
project. Let ``S`` be every committed prompt-and-output token of a request. At
any nonterminal block boundary:

* the target cache represents exactly ``S[:-1]``;
* an initialized draft cache represents ``S[:-1]`` as well;
* ``S[-1]`` is the *pending* token, not yet fed to any model.

Tentative suffixes may exist inside a block, during verification. They are never
present at a boundary.

This module owns the ledger of which tokens a cache represents and the crop
boundaries. It never copies a whole cache: a rollback is a length crop, which is
what makes a rejected suffix cheap to discard.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


class CacheError(RuntimeError):
    """A cache operation would violate the pending-token convention."""


@dataclass
class CacheHandle:
    """One model's KV cache plus the exact token prefix it represents.

    ``tokens`` is the ledger. It is the project's source of truth for cache
    length; the backend's own ``get_seq_length()`` is treated as something to
    *check against*, not to read state from, so a silent divergence between the
    two surfaces as an error instead of as wrong output.

    ``checks`` follows the engine's correctness mode. It is recorded in run
    artifacts, because enabling assertions for one engine and not another would
    make a timing comparison meaningless (CLAUDE.md).
    """

    name: str
    backend: Any = None
    tokens: list[int] = field(default_factory=list)
    checks: bool = True

    @property
    def length(self) -> int:
        """Number of token positions materialized in this cache."""
        return len(self.tokens)

    def extend(self, ids: Sequence[int]) -> None:
        """Record that ``ids`` were just appended by a forward pass."""
        if not len(ids):
            raise CacheError(f"{self.name}: refusing to record an empty extension")
        self.tokens.extend(int(value) for value in ids)

    def crop_to(self, length: int) -> None:
        """Drop everything after ``length`` positions.

        Growing is not a crop. A caller that wants more positions has to run a
        forward pass, because the keys and values for those positions do not
        exist yet; silently accepting a larger length would leave the ledger
        describing content the backend does not hold.
        """
        if length < 0:
            raise CacheError(f"{self.name}: crop length {length} is negative")
        if length > self.length:
            raise CacheError(
                f"{self.name}: cannot crop to {length}; cache holds only {self.length} positions"
            )
        del self.tokens[length:]

    def reset(self) -> None:
        """Forget everything. Used between requests so no state leaks (I10)."""
        self.tokens.clear()
        self.backend = None

    def represents(self, prefix: Sequence[int]) -> bool:
        """Does this cache hold exactly ``prefix``?"""
        return self.tokens == [int(value) for value in prefix]

    def assert_backend_agrees(self, backend_length: int | None) -> None:
        """The ledger and the backend must report the same number of positions."""
        if not self.checks or backend_length is None:
            return
        if backend_length != self.length:
            raise CacheError(
                f"{self.name}: ledger says {self.length} positions, backend says {backend_length}"
            )

    def assert_boundary(self, committed: Sequence[int]) -> None:
        """Enforce the pending-token convention against the committed sequence.

        ``committed`` is ``S``. The cache must hold exactly ``S[:-1]``: the same
        tokens, in the same order, one short. Checking token identity rather than
        only the length is what catches a stale tentative suffix that happens to
        have the right length -- the failure mode that produces plausible but
        wrong output (invariant I4).
        """
        if not self.checks:
            return
        if not committed:
            raise CacheError(f"{self.name}: boundary check needs at least one committed token")
        expected = [int(value) for value in committed[:-1]]
        if self.tokens != expected:
            raise CacheError(
                f"{self.name}: cache does not represent S[:-1] at a block boundary; "
                f"expected {len(expected)} tokens, hold {self.length}"
                + (
                    ""
                    if self.length != len(expected)
                    else f"; first difference at index {_first_difference(self.tokens, expected)}"
                )
            )


def _first_difference(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right))


def rollback_length(committed_before: int, accepted: int) -> int:
    """Cache length after committing ``accepted`` candidates plus one token.

    SPEC.md section 4.3 step 3: crop both caches to ``len(S) + r``, where ``S``
    is the sequence before the block and ``r`` is the number of accepted
    candidates. That equals ``len(S_new) - 1``, which is the pending-token
    convention restated, since the block commits ``r + 1`` tokens.
    """
    if committed_before < 1:
        raise CacheError(f"committed_before must be >= 1, got {committed_before}")
    if accepted < 0:
        raise CacheError(f"accepted must be >= 0, got {accepted}")
    return committed_before + accepted
