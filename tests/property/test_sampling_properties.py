"""Generated checks on the sampler and the oracle over the probability simplex."""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from switchback.oracle import (
    exact_fraction,
    rejection_step_distribution,
    residual,
)
from switchback.sampling import (
    ACCEPTANCE_STREAM,
    PROPOSAL_STREAM,
    STREAMS,
    TorchRandomSource,
    derive_stream_seed,
    greedy_token,
    normalize,
    positive_residual,
    softmax_probabilities,
)

# FP64 weights on the simplex. Values are kept well away from denormals so the
# exact-rational conversion below stays readable in a failing example.
weights = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=64),
    min_size=2,
    max_size=6,
)


@st.composite
def weight_pairs(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Two weight vectors of the *same* length.

    Generating them independently and filtering on equal length discards most
    inputs, which trips Hypothesis' filter_too_much health check intermittently.
    Drawing the shared size first removes the filter rather than suppressing the
    warning it produces.
    """
    size = draw(st.integers(min_value=2, max_value=6))
    element = st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=64
    )
    left = draw(st.lists(element, min_size=size, max_size=size))
    right = draw(st.lists(element, min_size=size, max_size=size))
    return left, right


logit_vectors = st.lists(
    st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False, width=32),
    min_size=2,
    max_size=8,
)
temperatures = st.floats(min_value=0.05, max_value=20.0, allow_nan=False, allow_infinity=False)


def exact_simplex(values: list[float]) -> tuple[Fraction, ...]:
    """Normalize FP64 weights into rationals that sum to exactly one."""
    fractions = [exact_fraction(value) for value in values]
    total = sum(fractions, Fraction(0))
    assume(total > 0)
    return tuple(value / total for value in fractions)


# --- oracle identity over the FP64 simplex ---------------------------------


@settings(max_examples=300)
@given(pair=weight_pairs())
def test_accepted_plus_residual_mass_equals_the_target(
    pair: tuple[list[float], list[float]],
) -> None:
    """The one-step identity, over generated FP64 distributions."""
    left, right = pair
    p = exact_simplex(left)
    q = exact_simplex(right)
    emitted = rejection_step_distribution(p, q)
    assert emitted == p
    for exact, approximate in zip(p, emitted, strict=True):
        assert abs(float(exact) - float(approximate)) < 1e-12


@settings(max_examples=200)
@given(pair=weight_pairs())
def test_residual_is_supported_where_the_target_exceeds_the_draft(
    pair: tuple[list[float], list[float]],
) -> None:
    left, right = pair
    p, q = exact_simplex(left), exact_simplex(right)
    assume(p != q)
    correction = residual(p, q)
    assert sum(correction, Fraction(0)) == 1
    for token, share in enumerate(correction):
        assert (share > 0) == (p[token] > q[token])


@settings(max_examples=200)
@given(values=weights)
def test_a_draft_equal_to_the_target_leaves_no_correction_mass(values: list[float]) -> None:
    p = exact_simplex(values)
    emitted = rejection_step_distribution(p, p)
    assert emitted == p


# --- production transforms -------------------------------------------------


@settings(max_examples=300)
@given(logits=logit_vectors, temperature=temperatures)
def test_softmax_produces_a_valid_distribution(logits: list[float], temperature: float) -> None:
    probabilities = softmax_probabilities(torch.tensor(logits, dtype=torch.float32), temperature)
    assert torch.isfinite(probabilities).all()
    assert bool((probabilities >= 0).all())
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)


@settings(max_examples=200)
@given(logits=logit_vectors, shift=st.floats(min_value=-50.0, max_value=50.0, width=32))
def test_softmax_is_shift_invariant(logits: list[float], shift: float) -> None:
    base = torch.tensor(logits, dtype=torch.float32)
    left = softmax_probabilities(base, 1.0)
    right = softmax_probabilities(base + shift, 1.0)
    assert torch.allclose(left, right, atol=1e-5)


@settings(max_examples=200)
@given(logits=logit_vectors)
def test_softmax_preserves_the_logit_ordering(logits: list[float]) -> None:
    base = torch.tensor(logits, dtype=torch.float32)
    probabilities = softmax_probabilities(base, 1.0)
    order = sorted(range(len(logits)), key=lambda i: (logits[i], -i))
    values = [float(probabilities[i]) for i in order]
    assert values == sorted(values)


@settings(max_examples=200)
@given(logits=logit_vectors)
def test_greedy_is_the_smallest_argmax(logits: list[float]) -> None:
    chosen = greedy_token(torch.tensor(logits, dtype=torch.float32))
    best = max(logits)
    assert logits[chosen] == best
    assert all(value < best for value in logits[:chosen])


@settings(max_examples=200)
@given(values=weights)
def test_normalize_is_idempotent(values: list[float]) -> None:
    tensor = torch.tensor(values, dtype=torch.float32)
    assume(float(tensor.sum()) > 1e-6)
    once = normalize(tensor)
    assert torch.allclose(once, normalize(once), atol=1e-6)


@settings(max_examples=200)
@given(pair=weight_pairs())
def test_positive_residual_never_goes_below_zero(
    pair: tuple[list[float], list[float]],
) -> None:
    left, right = pair
    p = torch.tensor(left, dtype=torch.float32)
    q = torch.tensor(right, dtype=torch.float32)
    assume(float(p.sum()) > 1e-6 and float(q.sum()) > 1e-6)
    correction = positive_residual(p / p.sum(), q / q.sum())
    assert bool((correction >= 0).all())
    assert torch.isfinite(correction).all()


# --- random streams --------------------------------------------------------


@settings(max_examples=100)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_stream_seeds_never_collide(seed: int) -> None:
    derived = [derive_stream_seed(seed, stream) for stream in STREAMS]
    assert len(set(derived)) == len(STREAMS)
    assert all(0 <= value < 2**63 for value in derived)


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, width=32),
        min_size=2,
        max_size=6,
    ),
)
def test_draws_only_land_on_supported_tokens(seed: int, values: list[float]) -> None:
    tensor = torch.tensor(values, dtype=torch.float32)
    assume(float(tensor.sum()) > 1e-3)
    source = TorchRandomSource(request_seed=seed)
    probabilities = tensor / tensor.sum()
    for _ in range(16):
        token = source.categorical(probabilities, PROPOSAL_STREAM)
        assert float(probabilities[token]) > 0.0


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_acceptance_is_certain_when_the_ratio_is_at_least_one(seed: int) -> None:
    source = TorchRandomSource(request_seed=seed)
    assert all(source.accept_ratio(0.5, 0.5, ACCEPTANCE_STREAM) for _ in range(16))
