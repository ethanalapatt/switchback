"""The pending-token convention and the crop boundaries that enforce it."""

from __future__ import annotations

import pytest

from switchback.cache import CacheError, CacheHandle, rollback_length


def handle(tokens: list[int], checks: bool = True) -> CacheHandle:
    return CacheHandle(name="target", backend=None, tokens=list(tokens), checks=checks)


def test_length_follows_the_ledger() -> None:
    cache = handle([])
    assert cache.length == 0
    cache.extend([1, 2, 3])
    assert cache.length == 3
    assert cache.represents([1, 2, 3])


def test_empty_extension_is_refused() -> None:
    with pytest.raises(CacheError, match="empty extension"):
        handle([1]).extend([])


def test_crop_drops_the_suffix() -> None:
    cache = handle([1, 2, 3, 4, 5])
    cache.crop_to(2)
    assert cache.tokens == [1, 2]


def test_crop_to_zero_is_allowed() -> None:
    cache = handle([1, 2])
    cache.crop_to(0)
    assert cache.length == 0


def test_crop_cannot_grow_the_cache() -> None:
    # Growing would leave the ledger describing keys and values that were never
    # computed, which is indistinguishable from a correct cache until the output
    # is wrong.
    with pytest.raises(CacheError, match="cannot crop to 9"):
        handle([1, 2, 3]).crop_to(9)


def test_negative_crop_is_refused() -> None:
    with pytest.raises(CacheError, match="negative"):
        handle([1, 2]).crop_to(-1)


def test_boundary_holds_when_the_cache_is_one_token_short() -> None:
    handle([1, 2, 3]).assert_boundary([1, 2, 3, 4])


def test_boundary_rejects_a_cache_that_is_too_long() -> None:
    with pytest.raises(CacheError, match="S\\[:-1\\]"):
        handle([1, 2, 3, 4]).assert_boundary([1, 2, 3, 4])


def test_boundary_rejects_a_cache_that_is_too_short() -> None:
    with pytest.raises(CacheError, match="S\\[:-1\\]"):
        handle([1, 2]).assert_boundary([1, 2, 3, 4])


def test_boundary_rejects_a_stale_suffix_of_the_right_length() -> None:
    """The dangerous case: correct length, wrong keys (invariant I4)."""
    with pytest.raises(CacheError, match="first difference at index 2"):
        handle([1, 2, 99]).assert_boundary([1, 2, 3, 4])


def test_boundary_needs_at_least_one_committed_token() -> None:
    with pytest.raises(CacheError, match="at least one committed token"):
        handle([]).assert_boundary([])


def test_checks_can_be_disabled_but_the_ledger_still_tracks() -> None:
    cache = handle([1, 2, 99], checks=False)
    cache.assert_boundary([1, 2, 3, 4])  # no raise
    cache.assert_backend_agrees(17)  # no raise
    assert cache.length == 3


def test_backend_disagreement_is_an_error() -> None:
    with pytest.raises(CacheError, match="backend says 5"):
        handle([1, 2, 3]).assert_backend_agrees(5)


def test_backend_length_none_is_tolerated() -> None:
    handle([1, 2, 3]).assert_backend_agrees(None)


def test_reset_clears_state_between_requests() -> None:
    cache = handle([1, 2, 3])
    cache.backend = object()
    cache.reset()
    assert cache.length == 0
    assert cache.backend is None


def test_rollback_length_restates_the_pending_token_convention() -> None:
    # After a block starting from |S| = 5 accepts 2 candidates, the request
    # commits 3 tokens, so |S_new| = 8 and the cache must hold 7 positions.
    assert rollback_length(5, 2) == 7 == (5 + 2 + 1) - 1
    assert rollback_length(5, 0) == 5
    assert rollback_length(1, 0) == 1


def test_rollback_length_rejects_impossible_inputs() -> None:
    with pytest.raises(CacheError, match="committed_before"):
        rollback_length(0, 1)
    with pytest.raises(CacheError, match="accepted"):
        rollback_length(5, -1)
