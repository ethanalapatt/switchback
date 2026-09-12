"""Probability transforms, greedy tie breaking, residuals, and block structure."""

from __future__ import annotations

import math

import pytest
import torch

from switchback.sampling import (
    ACCEPTANCE_STREAM,
    CORRECTION_STREAM,
    PROPOSAL_STREAM,
    STREAMS,
    BlockOutcome,
    TorchRandomSource,
    derive_stream_seed,
    forbidden_token_ids,
    greedy_block,
    greedy_sequence,
    greedy_target_sequence,
    greedy_token,
    normalize,
    positive_residual,
    softmax_probabilities,
    speculative_block,
    validate_distribution,
)
from switchback.types import DecodeConfig, NumericalSamplingError


def vector(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def constant_row(probabilities: torch.Tensor):
    return lambda prefix: probabilities


# --- probability transforms ------------------------------------------------


def test_softmax_matches_a_hand_computed_reference() -> None:
    probabilities = softmax_probabilities(vector(1.0, 2.0, 3.0), temperature=1.0)
    expected = [math.exp(v - 3.0) for v in (1.0, 2.0, 3.0)]
    total = sum(expected)
    for index, value in enumerate(expected):
        assert float(probabilities[index]) == pytest.approx(value / total, abs=1e-7)


def test_softmax_is_stable_for_large_logits() -> None:
    # Without the max subtraction this overflows to inf/inf = nan.
    probabilities = softmax_probabilities(vector(1000.0, 1001.0, 999.0), temperature=1.0)
    assert torch.isfinite(probabilities).all()
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-6)
    assert int(probabilities.argmax()) == 1


def test_temperature_sharpens_and_flattens() -> None:
    logits = vector(0.0, 1.0, 2.0)
    cold = softmax_probabilities(logits, temperature=0.1)
    hot = softmax_probabilities(logits, temperature=10.0)
    assert float(cold.max()) > float(hot.max())
    assert float(hot.min()) > float(cold.min())


def test_suppressed_tokens_are_removed_before_renormalization() -> None:
    # Masking after the softmax would leave the remaining mass below one.
    probabilities = softmax_probabilities(vector(1.0, 2.0, 3.0), 1.0, forbidden_ids=[2])
    assert float(probabilities[2]) == 0.0
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-6)
    unmasked = softmax_probabilities(vector(1.0, 2.0), 1.0)
    assert float(probabilities[0]) == pytest.approx(float(unmasked[0]), abs=1e-7)


def test_nonfinite_logits_are_rejected() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(NumericalSamplingError, match="nonfinite"):
            softmax_probabilities(vector(1.0, bad, 3.0), 1.0)


def test_non_positive_temperature_is_rejected() -> None:
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(NumericalSamplingError, match="temperature"):
            softmax_probabilities(vector(1.0, 2.0), bad)


def test_suppressing_every_token_is_an_error_not_a_uniform_fallback() -> None:
    with pytest.raises(NumericalSamplingError, match="no support"):
        softmax_probabilities(vector(1.0, 2.0), 1.0, forbidden_ids=[0, 1])


def test_distribution_validation_catches_each_defect() -> None:
    with pytest.raises(NumericalSamplingError, match="nonfinite"):
        validate_distribution(vector(0.5, float("nan")), "d")
    with pytest.raises(NumericalSamplingError, match="negative"):
        validate_distribution(vector(1.5, -0.5), "d")
    with pytest.raises(NumericalSamplingError, match="zero total mass"):
        validate_distribution(vector(0.0, 0.0), "d")


def test_normalize_accepts_unnormalized_weights() -> None:
    probabilities = normalize(vector(1.0, 3.0))
    assert float(probabilities[0]) == pytest.approx(0.25)
    assert float(probabilities.sum()) == pytest.approx(1.0)


# --- greedy ----------------------------------------------------------------


def test_greedy_breaks_ties_towards_the_smallest_token_id() -> None:
    assert greedy_token(vector(3.0, 3.0, 1.0)) == 0
    assert greedy_token(vector(1.0, 3.0, 3.0)) == 1
    assert greedy_token(vector(2.0, 2.0, 2.0)) == 0


def test_greedy_respects_suppressed_tokens() -> None:
    assert greedy_token(vector(5.0, 1.0, 2.0), forbidden_ids=[0]) == 2


def test_greedy_rejects_nonfinite_logits() -> None:
    with pytest.raises(NumericalSamplingError, match="nonfinite"):
        greedy_token(vector(1.0, float("nan")))


# --- residual --------------------------------------------------------------


def test_residual_keeps_only_the_positive_part() -> None:
    residual = positive_residual(vector(0.25, 0.25, 0.5), vector(0.5, 0.25, 0.25))
    assert residual.tolist() == [0.0, 0.0, 0.25]


def test_residual_of_identical_distributions_is_zero() -> None:
    p = vector(0.25, 0.25, 0.5)
    assert float(positive_residual(p, p).sum()) == 0.0


def test_residual_of_disjoint_supports_is_the_target() -> None:
    residual = positive_residual(vector(0.0, 1.0), vector(1.0, 0.0))
    assert residual.tolist() == [0.0, 1.0]


def test_residual_rejects_mismatched_sizes() -> None:
    with pytest.raises(NumericalSamplingError, match="differ in shape"):
        positive_residual(vector(0.5, 0.5), vector(0.25, 0.25, 0.5))


# --- random source ---------------------------------------------------------


def test_stream_seeds_are_distinct_and_reproducible() -> None:
    seeds = {stream: derive_stream_seed(42, stream) for stream in STREAMS}
    assert len(set(seeds.values())) == len(STREAMS)
    assert seeds == {stream: derive_stream_seed(42, stream) for stream in STREAMS}
    assert derive_stream_seed(42, PROPOSAL_STREAM) != derive_stream_seed(43, PROPOSAL_STREAM)


def test_unknown_stream_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown stream"):
        derive_stream_seed(42, "bonus")


def test_streams_advance_independently() -> None:
    # Consuming the proposal stream must not change what the acceptance stream
    # returns next; otherwise a controller changing the draft length would
    # perturb acceptance decisions (invariant I8).
    reference = TorchRandomSource(request_seed=7)
    baseline = [reference.accept_ratio(0.5, 1.0, ACCEPTANCE_STREAM) for _ in range(8)]

    perturbed = TorchRandomSource(request_seed=7)
    for _ in range(5):
        perturbed.categorical(vector(0.5, 0.5), PROPOSAL_STREAM)
    assert [perturbed.accept_ratio(0.5, 1.0, ACCEPTANCE_STREAM) for _ in range(8)] == baseline


def test_same_seed_reproduces_the_same_draws() -> None:
    weights = vector(0.1, 0.2, 0.7)
    left = TorchRandomSource(request_seed=99)
    right = TorchRandomSource(request_seed=99)
    assert [left.categorical(weights, CORRECTION_STREAM) for _ in range(32)] == [
        right.categorical(weights, CORRECTION_STREAM) for _ in range(32)
    ]


def test_categorical_never_returns_an_unsupported_token() -> None:
    source = TorchRandomSource(request_seed=5)
    weights = vector(0.0, 1.0, 0.0, 0.0)
    assert {source.categorical(weights, PROPOSAL_STREAM) for _ in range(64)} == {1}


def test_accept_ratio_is_certain_when_the_target_dominates() -> None:
    source = TorchRandomSource(request_seed=5)
    assert all(source.accept_ratio(0.8, 0.4, ACCEPTANCE_STREAM) for _ in range(64))


def test_accept_ratio_rejects_a_zero_denominator() -> None:
    source = TorchRandomSource(request_seed=5)
    with pytest.raises(NumericalSamplingError, match="no support"):
        source.accept_ratio(0.5, 0.0, ACCEPTANCE_STREAM)


# --- block structure -------------------------------------------------------


def test_block_outcome_enforces_its_own_invariants() -> None:
    with pytest.raises(AssertionError, match="emitted"):
        BlockOutcome(
            proposed=(1, 2), accepted=2, rejection_position=None, emitted=(1,), all_accepted=True
        )
    with pytest.raises(AssertionError, match="all_accepted"):
        BlockOutcome(
            proposed=(1,), accepted=1, rejection_position=None, emitted=(1, 2), all_accepted=False
        )
    with pytest.raises(AssertionError, match="rejection position"):
        BlockOutcome(
            proposed=(1, 2), accepted=0, rejection_position=1, emitted=(5,), all_accepted=False
        )


def test_identical_distributions_accept_everything() -> None:
    row = vector(0.25, 0.25, 0.5)
    outcome = speculative_block((), 4, constant_row(row), constant_row(row), TorchRandomSource(1))
    assert outcome.all_accepted
    assert outcome.accepted == 4
    assert len(outcome.emitted) == 5
    assert outcome.emitted[:4] == outcome.proposed


def test_disjoint_distributions_reject_the_first_candidate() -> None:
    target = vector(0.0, 1.0)
    draft = vector(1.0, 0.0)
    outcome = speculative_block(
        (), 3, constant_row(target), constant_row(draft), TorchRandomSource(1)
    )
    assert outcome.accepted == 0
    assert outcome.rejection_position == 0
    # The correction must come from the residual, which here is the target.
    assert outcome.emitted == (1,)


def test_block_requires_a_positive_length() -> None:
    row = vector(0.5, 0.5)
    with pytest.raises(ValueError, match="gamma >= 1"):
        speculative_block((), 0, constant_row(row), constant_row(row), TorchRandomSource(1))


def test_rows_are_queried_at_the_intended_causal_prefixes() -> None:
    """Row j must be conditioned on the prefix plus the first j candidates."""
    seen: list[tuple[int, ...]] = []

    def target_row(prefix: tuple[int, ...]) -> torch.Tensor:
        seen.append(prefix)
        return vector(0.5, 0.5)

    outcome = speculative_block(
        (7, 8), 3, target_row, constant_row(vector(0.5, 0.5)), TorchRandomSource(3)
    )
    expected = [(7, 8)]
    for index in range(outcome.accepted):
        expected.append((7, 8, *outcome.proposed[: index + 1]))
    assert seen == expected


def test_zero_probability_proposal_is_refused() -> None:
    class BadSource:
        def categorical(self, weights: torch.Tensor, stream: str) -> int:
            return 0  # index 0 has no mass under the draft below

        def accept_ratio(self, numerator: float, denominator: float, stream: str) -> bool:
            return True

    with pytest.raises(NumericalSamplingError, match="zero probability"):
        speculative_block(
            (), 1, constant_row(vector(0.5, 0.5)), constant_row(vector(0.0, 1.0)), BadSource()
        )


def test_zero_residual_after_rejection_raises_rather_than_substituting() -> None:
    class AlwaysReject:
        def categorical(self, weights: torch.Tensor, stream: str) -> int:
            return int(torch.nonzero(weights > 0).flatten()[0])

        def accept_ratio(self, numerator: float, denominator: float, stream: str) -> bool:
            return False

    row = vector(0.25, 0.75)
    with pytest.raises(NumericalSamplingError, match="residual"):
        speculative_block((), 1, constant_row(row), constant_row(row), AlwaysReject())


# --- greedy blocks ---------------------------------------------------------


def test_greedy_block_accepts_a_matching_draft() -> None:
    row = vector(0.1, 0.9)
    outcome = greedy_block((), 3, constant_row(row), constant_row(row))
    assert outcome.all_accepted
    assert outcome.emitted == (1, 1, 1, 1)


def test_greedy_block_corrects_at_the_first_disagreement() -> None:
    outcome = greedy_block((), 3, constant_row(vector(0.9, 0.1)), constant_row(vector(0.1, 0.9)))
    assert outcome.accepted == 0
    assert outcome.rejection_position == 0
    assert outcome.emitted == (0,)


def test_greedy_speculation_equals_greedy_target_only() -> None:
    """Invariant I6 in miniature, on a tree where the draft is often wrong."""
    torch.manual_seed(0)
    rows = {}

    def make(prefix: tuple[int, ...], shift: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(hash((prefix, shift)) % (2**31))
        return torch.softmax(torch.randn(4, generator=generator), dim=-1)

    def target_row(prefix: tuple[int, ...]) -> torch.Tensor:
        return rows.setdefault(("t", prefix), make(prefix, 0))

    def draft_row(prefix: tuple[int, ...]) -> torch.Tensor:
        return rows.setdefault(("d", prefix), make(prefix, 1))

    reference = greedy_target_sequence((), 12, target_row)
    for gamma in (1, 2, 4, 8):
        assert greedy_sequence((), gamma, 12, target_row, draft_row) == reference


# --- EOS policy ------------------------------------------------------------


def test_forbidden_ids_follow_the_eos_policy() -> None:
    suppress = DecodeConfig("sample", 0.7, 32, "suppress_until_budget", 1)
    respect = DecodeConfig("sample", 0.7, 32, "respect", 1)
    assert forbidden_token_ids(suppress, [151645, 151643]) == (151643, 151645)
    assert forbidden_token_ids(respect, [151645, 151643]) == ()
