"""Independent exact oracle for speculative sampling.

This module must never import :mod:`switchback.sampling` or call its acceptance,
residual, or draw functions (SPEC.md section 3). It re-derives the algorithm's
output distribution from the specification using :class:`fractions.Fraction`, so
"exact" means exact: no floating point appears in a reference value.

It provides three things.

``target_sequence_distribution``
    The distribution the target model alone would produce over ``n``-token
    continuations. This is the answer speculative decoding has to reproduce.

``rejection_step_distribution``
    One speculative step's output distribution, obtained by enumerating the
    algorithm's branches (candidate draw x accept/reject x residual draw) and
    summing their exact masses.

``enumerate_algorithm``
    A driver that runs *another* implementation over every execution path,
    branching at each random decision and weighting by its exact probability.
    The implementation under test is passed in as a callable, so the oracle
    still does not import it. This is how the production sampler is checked
    exhaustively rather than statistically.

A note on scope: these are finite-model checks. They establish that the
algorithm and its implementation agree with the target distribution on small
enumerable cases. They are not a proof of equivalence for every real model under
every GPU kernel, and they say nothing about floating-point behaviour at
vocabulary size 151,936. See ``docs/correctness.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

Rational = Fraction
Distribution = tuple[Fraction, ...]
"""Probabilities over a vocabulary ``[V]``, exact and summing to one."""

Prefix = tuple[int, ...]


class OracleError(ValueError):
    """An oracle input was not a valid exact distribution."""


def exact_fraction(value: object) -> Fraction:
    """Convert a scalar to an exact Fraction, including a 0-dim torch tensor.

    ``Fraction(float)`` is exact for the float's actual value, so a dyadic
    rational that survived FP32 arithmetic converts without loss. A Fraction or
    int passes through untouched, which keeps a pure-oracle computation exact
    even where the value is not representable in binary floating point.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value)
    item = getattr(value, "item", None)
    if callable(item):
        return Fraction(float(item()))
    raise OracleError(f"cannot convert {value!r} to an exact fraction")


def as_distribution(values: Iterable[object], name: str = "distribution") -> Distribution:
    """Coerce exact-representable values to a validated rational distribution."""
    probabilities = tuple(exact_fraction(value) for value in values)
    if not probabilities:
        raise OracleError(f"{name} is empty")
    if any(value < 0 for value in probabilities):
        raise OracleError(f"{name} contains a negative probability")
    total = sum(probabilities, Fraction(0))
    if total != 1:
        raise OracleError(f"{name} sums to {total}, not 1")
    return probabilities


def normalized(weights: Sequence[Fraction], name: str = "weights") -> Distribution:
    """Exactly normalize non-negative weights."""
    total = sum(weights, Fraction(0))
    if total <= 0:
        raise OracleError(f"{name} has zero total mass")
    return tuple(weight / total for weight in weights)


def residual(target: Distribution, draft: Distribution) -> Distribution:
    """The normalized positive residual ``max(p - x, 0) / sum`` used on rejection.

    Derived here from the specification, not read from the production module.
    """
    if len(target) != len(draft):
        raise OracleError("target and draft distributions have different sizes")
    positive = [max(p - q, Fraction(0)) for p, q in zip(target, draft, strict=True)]
    return normalized(positive, "positive residual")


def rejection_step_distribution(target: Distribution, draft: Distribution) -> Distribution:
    """Enumerate one speculative step and return its exact output distribution.

    The enumeration is over the algorithm's own branches:

    * candidate ``d`` is drawn with probability ``q(d)``;
    * it is accepted with probability ``min(1, p(d)/q(d))`` and emitted;
    * otherwise a correction is drawn from the normalized positive residual.

    In exact arithmetic the accepted mass at ``x`` is ``min(p(x), q(x))`` and the
    rejected mass is ``1 - sum_y min(p(y), q(y))``, which is exactly the
    normalizer of the positive residual, so the two contributions sum to ``p``.
    That identity is *not* assumed below: the branches are summed and the caller
    compares the result against ``target``.
    """
    if len(target) != len(draft):
        raise OracleError("target and draft distributions have different sizes")
    size = len(target)
    emitted = [Fraction(0) for _ in range(size)]
    rejected_mass = Fraction(0)

    for candidate in range(size):
        q = draft[candidate]
        if q == 0:
            # A candidate the draft can never propose contributes nothing. It is
            # not an error: zero support on the draft side is a tested case.
            continue
        accept = min(Fraction(1), target[candidate] / q)
        emitted[candidate] += q * accept
        rejected_mass += q * (1 - accept)

    if rejected_mass > 0:
        for token, share in enumerate(residual(target, draft)):
            emitted[token] += rejected_mass * share
    return tuple(emitted)


@dataclass(frozen=True)
class Tree:
    """A tiny explicit model: a next-token distribution for each reachable prefix.

    Small enough to enumerate exhaustively. ``rows`` maps a prefix tuple to a
    distribution; ``vocab_size`` fixes the alphabet. A prefix that is missing
    when queried is an error, not a silently uniform fallback.
    """

    vocab_size: int
    rows: Mapping[Prefix, Distribution]

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise OracleError(f"vocab_size must be >= 1, got {self.vocab_size}")
        for prefix, row in self.rows.items():
            if len(row) != self.vocab_size:
                raise OracleError(
                    f"row at {prefix} has {len(row)} entries, expected {self.vocab_size}"
                )
            as_distribution(row, f"row at {prefix}")

    def row(self, prefix: Prefix) -> Distribution:
        try:
            return self.rows[prefix]
        except KeyError as error:
            raise OracleError(f"tree has no row for prefix {prefix}") from error


def target_sequence_distribution(tree: Tree, prefix: Prefix, length: int) -> dict[Prefix, Fraction]:
    """Exact distribution over ``length``-token continuations under the target.

    This is plain autoregressive decoding: the product of conditional
    probabilities along each path. It is the reference every speculative
    configuration must match.
    """
    if length < 0:
        raise OracleError(f"length must be >= 0, got {length}")
    distribution: dict[Prefix, Fraction] = {(): Fraction(1)}
    for _ in range(length):
        expanded: dict[Prefix, Fraction] = {}
        for suffix, mass in distribution.items():
            row = tree.row(prefix + suffix)
            for token, probability in enumerate(row):
                if probability == 0:
                    continue
                expanded[(*suffix, token)] = expanded.get((*suffix, token), Fraction(0)) + (
                    mass * probability
                )
        distribution = expanded
    return distribution


# --- Exhaustive driver for an external implementation ----------------------


class _Branch(Exception):
    """Internal control signal: this execution path is finished."""


@dataclass
class EnumeratingSource:
    """A random source that replays one predetermined execution path.

    ``decisions`` is the sequence of outcomes to return, and ``trace`` records
    the exact probability of each one so the driver can weight the path. When
    the script runs out, :class:`_Branch` is raised to signal the driver that the
    path must be extended.
    """

    decisions: tuple[int, ...]
    position: int = 0
    weight: Fraction = Fraction(1)
    pending: tuple[str, tuple[Fraction, ...]] | None = None

    def _next(self, name: str, outcomes: tuple[Fraction, ...]) -> int:
        if self.position >= len(self.decisions):
            self.pending = (name, outcomes)
            raise _Branch
        choice = self.decisions[self.position]
        self.position += 1
        self.weight *= outcomes[choice]
        return choice

    def categorical(self, weights: Sequence[object], stream: str) -> int:
        exact = normalized([exact_fraction(value) for value in weights], "categorical weights")
        return self._next(f"categorical/{stream}", exact)

    def accept_ratio(self, numerator: object, denominator: object, stream: str) -> bool:
        num, den = exact_fraction(numerator), exact_fraction(denominator)
        if den <= 0:
            raise OracleError("acceptance ratio has a non-positive denominator")
        accept = min(Fraction(1), num / den)
        # Index 0 is reject, index 1 is accept, so the outcome tuple lines up
        # with bool(choice).
        return bool(self._next(f"accept/{stream}", (1 - accept, accept)))


def enumerate_algorithm(
    run: Callable[[EnumeratingSource], object],
    max_paths: int = 100_000,
) -> dict[object, Fraction]:
    """Run ``run`` over every execution path and return its exact output law.

    ``run`` receives a source and returns a hashable result. The driver performs
    a depth-first walk: it replays a decision prefix, and when the callable asks
    for a decision the prefix does not cover, it branches over every outcome with
    non-zero probability. Zero-probability outcomes are skipped, so an impossible
    branch never appears in the result.

    The implementation under test is a parameter. The oracle never imports it.
    """
    outcomes: dict[object, Fraction] = {}
    stack: list[tuple[int, ...]] = [()]
    visited = 0
    while stack:
        script = stack.pop()
        visited += 1
        if visited > max_paths:
            raise OracleError(
                f"enumeration exceeded {max_paths} paths; shrink the case rather than raising this"
            )
        source = EnumeratingSource(decisions=script)
        try:
            result = run(source)
        except _Branch:
            assert source.pending is not None
            _, probabilities = source.pending
            for choice, probability in enumerate(probabilities):
                if probability > 0:
                    stack.append((*script, choice))
            continue
        if source.position != len(script):
            raise OracleError("implementation consumed fewer decisions than the script supplied")
        key = _hashable(result)
        outcomes[key] = outcomes.get(key, Fraction(0)) + source.weight
    return outcomes


def _hashable(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def total_variation_distance(
    left: Mapping[object, Fraction], right: Mapping[object, Fraction]
) -> Fraction:
    """Exact total variation distance between two finite distributions."""
    keys = set(left) | set(right)
    difference = sum(abs(left.get(key, Fraction(0)) - right.get(key, Fraction(0))) for key in keys)
    return Fraction(difference, 2)
