"""Exact enumeration checks against an independent oracle.

These are finite-model checks. They show the algorithm and this implementation
agree with the target distribution on small enumerable cases. They are not a
proof for a real model at vocabulary size 151,936, and they are not a claim that
two engines produce identical strings from the same seed.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
import torch

from switchback import oracle
from switchback.oracle import (
    OracleError,
    Tree,
    as_distribution,
    enumerate_algorithm,
    exact_fraction,
    rejection_step_distribution,
    residual,
    target_sequence_distribution,
    total_variation_distance,
)
from switchback.sampling import speculative_block, speculative_sequence

DENOMINATOR = 4


def compositions(total: int, parts: int) -> list[tuple[Fraction, ...]]:
    """Every rational distribution over ``parts`` tokens with denominator ``total``."""
    if parts == 1:
        return [(Fraction(total, DENOMINATOR),)]
    out: list[tuple[Fraction, ...]] = []
    for head in range(total + 1):
        for tail in compositions(total - head, parts - 1):
            out.append((Fraction(head, DENOMINATOR), *tail))
    return out


def to_tensor(distribution: tuple[Fraction, ...]) -> torch.Tensor:
    """Dyadic rationals with denominator 4 are exact in FP32, so this is lossless."""
    return torch.tensor([float(value) for value in distribution], dtype=torch.float32)


def constant(row: torch.Tensor):
    return lambda prefix: row


# --- the oracle's own derivation ------------------------------------------


@pytest.mark.parametrize("size", [2, 3, 4])
def test_accepted_plus_residual_mass_equals_the_target_exactly(size: int) -> None:
    """SPEC.md M2 acceptance: vocabularies 2-4, probability denominator 4.

    Includes zero support, identical, and disjoint distributions because
    ``compositions`` enumerates every such pair rather than sampling them.
    """
    population = compositions(DENOMINATOR, size)
    checked = 0
    for target in population:
        p = as_distribution(target)
        for draft in population:
            q = as_distribution(draft)
            emitted = rejection_step_distribution(p, q)
            assert emitted == p, (target, draft, emitted)
            # The stated gate is 1e-12 after conversion to FP64; equality above
            # is stronger, so this only confirms the conversion is well behaved.
            for exact, approximate in zip(p, emitted, strict=True):
                assert abs(float(exact) - float(approximate)) < 1e-12
            checked += 1
    assert checked == len(population) ** 2


def test_identical_distributions_have_no_residual() -> None:
    p = as_distribution([Fraction(1, 4), Fraction(3, 4)])
    with pytest.raises(OracleError, match="zero total mass"):
        residual(p, p)


def test_disjoint_distributions_put_all_residual_mass_on_the_target() -> None:
    p = as_distribution([Fraction(0), Fraction(1)])
    q = as_distribution([Fraction(1), Fraction(0)])
    assert residual(p, q) == p
    assert rejection_step_distribution(p, q) == p


def test_a_draft_with_zero_support_on_a_target_token_still_reproduces_the_target() -> None:
    p = as_distribution([Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)])
    q = as_distribution([Fraction(1, 2), Fraction(1, 2), Fraction(0)])
    assert rejection_step_distribution(p, q) == p


def test_invalid_distributions_are_refused() -> None:
    with pytest.raises(OracleError, match="sums to"):
        as_distribution([Fraction(1, 2), Fraction(1, 4)])
    with pytest.raises(OracleError, match="negative"):
        as_distribution([Fraction(3, 2), Fraction(-1, 2)])
    with pytest.raises(OracleError, match="empty"):
        as_distribution([])


def test_exact_fraction_round_trips_a_float32_dyadic() -> None:
    tensor = torch.tensor([0.25, 0.75], dtype=torch.float32)
    assert [exact_fraction(value) for value in tensor] == [Fraction(1, 4), Fraction(3, 4)]


# --- the production sampler, enumerated exhaustively -----------------------


@pytest.mark.parametrize("size", [2, 3])
def test_production_block_reproduces_the_target_for_every_rational_pair(size: int) -> None:
    """Drive the real sampler over every execution path and compare exactly."""
    population = compositions(DENOMINATOR, size)
    for target in population:
        p = as_distribution(target)
        p_tensor = to_tensor(target)
        for draft in population:
            q_tensor = to_tensor(draft)
            law = enumerate_algorithm(
                lambda source, a=p_tensor, b=q_tensor: speculative_block(
                    (), 1, constant(a), constant(b), source
                )
            )
            first: dict[int, Fraction] = {}
            for outcome, mass in law.items():
                token = outcome.emitted[0]  # type: ignore[attr-defined]
                first[token] = first.get(token, Fraction(0)) + mass
            assert sum(law.values()) == 1
            for token, probability in enumerate(p):
                assert first.get(token, Fraction(0)) == probability, (target, draft, token)


def test_rejection_happens_at_every_position_across_the_enumerated_paths() -> None:
    """A block of length 4 must exercise rejection at 0..3 and full acceptance."""
    p = to_tensor((Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)))
    q = to_tensor((Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)))
    law = enumerate_algorithm(
        lambda source: speculative_block((), 4, constant(p), constant(q), source)
    )
    positions = {outcome.rejection_position for outcome in law}  # type: ignore[attr-defined]
    assert positions == {0, 1, 2, 3, None}
    for outcome, mass in law.items():
        assert mass > 0
        assert len(outcome.emitted) == outcome.accepted + 1  # type: ignore[attr-defined]


def test_block_length_does_not_change_the_first_token_distribution() -> None:
    p = as_distribution((Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)))
    p_tensor = to_tensor(tuple(p))
    q_tensor = to_tensor((Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)))
    for gamma in (1, 2, 3):
        law = enumerate_algorithm(
            lambda source, g=gamma: speculative_block(
                (), g, constant(p_tensor), constant(q_tensor), source
            )
        )
        first: dict[int, Fraction] = {}
        for outcome, mass in law.items():
            token = outcome.emitted[0]  # type: ignore[attr-defined]
            first[token] = first.get(token, Fraction(0)) + mass
        assert tuple(first[token] for token in range(3)) == p


# --- multi-step trees ------------------------------------------------------


def random_tree(rng: random.Random, vocab_size: int, depth: int) -> Tree:
    """A tiny tree with denominator-4 rational rows for every prefix up to ``depth``."""
    rows: dict[tuple[int, ...], tuple[Fraction, ...]] = {}
    frontier: list[tuple[int, ...]] = [()]
    for _ in range(depth + 1):
        following: list[tuple[int, ...]] = []
        for prefix in frontier:
            rows[prefix] = tuple(rng.choice(compositions(DENOMINATOR, vocab_size)))
            following.extend((*prefix, token) for token in range(vocab_size))
        frontier = following
    return Tree(vocab_size=vocab_size, rows=rows)


def tree_rows(tree: Tree):
    cache: dict[tuple[int, ...], torch.Tensor] = {}

    def row(prefix: tuple[int, ...]) -> torch.Tensor:
        if prefix not in cache:
            cache[prefix] = to_tensor(tuple(tree.row(prefix)))
        return cache[prefix]

    return row


@pytest.mark.parametrize("case", range(20))
def test_twenty_seeded_tree_pairs_preserve_the_target_distribution(case: int) -> None:
    """SPEC.md M2: 20 deterministic tiny tree pairs, vocab 2-3, <= 3 output tokens.

    Enumerating outcomes *within* each tree pair catches prefix-conditioning
    mistakes -- for example verifying candidate j against the row for the
    original prefix instead of prefix+candidates[:j] -- without enumerating all
    possible model trees.
    """
    rng = random.Random(1000 + case)
    vocab_size = 2 + (case % 2)
    length = 3
    target = random_tree(rng, vocab_size, depth=length)
    draft = random_tree(rng, vocab_size, depth=length)

    expected = target_sequence_distribution(target, (), length)
    target_row, draft_row = tree_rows(target), tree_rows(draft)

    for gamma in (1, 2):
        law = enumerate_algorithm(
            lambda source, g=gamma: speculative_sequence(
                (), g, length, target_row, draft_row, source
            )
        )
        assert sum(law.values()) == 1
        assert all(len(sequence) == length for sequence in law)  # type: ignore[arg-type]
        assert total_variation_distance(law, expected) == 0, (case, gamma)


def test_a_prefix_conditioning_mistake_is_actually_detected() -> None:
    """The tree test must fail for a wrong implementation, or it proves nothing.

    This deliberately verifies every candidate against the block's first target
    row, which is the classic off-by-one in speculative verification.
    """
    rng = random.Random(4242)
    target = random_tree(rng, 2, depth=3)
    draft = random_tree(rng, 2, depth=3)
    target_row, draft_row = tree_rows(target), tree_rows(draft)
    expected = target_sequence_distribution(target, (), 3)

    def wrong_target_row(prefix: tuple[int, ...]) -> torch.Tensor:
        return target_row(prefix[:1] if len(prefix) > 1 else prefix)

    law = enumerate_algorithm(
        lambda source: speculative_sequence((), 2, 3, wrong_target_row, draft_row, source)
    )
    assert total_variation_distance(law, expected) > 0


def test_target_sequence_distribution_is_a_product_of_conditionals() -> None:
    tree = Tree(
        vocab_size=2,
        rows={
            (): as_distribution([Fraction(1, 4), Fraction(3, 4)]),
            (0,): as_distribution([Fraction(1, 2), Fraction(1, 2)]),
            (1,): as_distribution([Fraction(1), Fraction(0)]),
        },
    )
    law = target_sequence_distribution(tree, (), 2)
    assert law[(0, 0)] == Fraction(1, 8)
    assert law[(0, 1)] == Fraction(1, 8)
    assert law[(1, 0)] == Fraction(3, 4)
    assert (1, 1) not in law  # zero-probability paths are dropped, not stored
    assert sum(law.values()) == 1


def test_tree_refuses_a_missing_row_instead_of_guessing() -> None:
    tree = Tree(vocab_size=2, rows={(): as_distribution([Fraction(1, 2), Fraction(1, 2)])})
    with pytest.raises(OracleError, match="no row for prefix"):
        tree.row((0,))


def test_malformed_tree_rows_are_refused() -> None:
    with pytest.raises(OracleError, match="expected 3"):
        Tree(vocab_size=3, rows={(): as_distribution([Fraction(1, 2), Fraction(1, 2)])})


# --- independence ----------------------------------------------------------


def test_the_oracle_does_not_import_the_production_sampler() -> None:
    """SPEC.md section 3: oracle.py must not call production acceptance code.

    Checked by parsing the module's imports rather than by searching its text,
    so the prose that explains the rule cannot satisfy or break the check.
    """
    import ast
    from pathlib import Path as _Path

    tree = ast.parse(_Path(oracle.__file__ or "").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)
    forbidden = {name for name in imported if "sampling" in name or name == "torch"}
    assert forbidden == set(), forbidden
    assert not hasattr(oracle, "speculative_block")
    assert not hasattr(oracle, "positive_residual")


def test_the_enumerator_refuses_a_runaway_case() -> None:
    row = to_tensor(tuple(compositions(DENOMINATOR, 4)[0]))
    uniform = torch.tensor([0.25] * 4, dtype=torch.float32)
    with pytest.raises(OracleError, match="exceeded"):
        enumerate_algorithm(
            lambda source: speculative_sequence((), 8, 8, constant(uniform), constant(row), source),
            max_paths=500,
        )
