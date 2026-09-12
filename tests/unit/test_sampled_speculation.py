"""The sampled decoder, enumerated exhaustively against the independent oracle.

The decoder is driven through its *real* path here: scripted adapters with KV
ledgers, the softmax transform, the cache crops and the catch-up call. Only the
model weights are replaced. That is the difference from
``tests/unit/test_oracle.py``, which enumerates the pure sampler in isolation.

Exactness is preserved on purpose. A row is uniform over a subset of a 4-token
vocabulary, and the masked tokens carry a logit of -1000 which exponentiates to
exactly 0.0 in FP32. So every probability the decoder computes is 1, 1/2 or 1/4,
all dyadic, and the residual, the acceptance ratio and the enumerated branch
weights are exact rationals. No tolerance is needed and none is used.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction

import pytest
import torch

from switchback.cache import CacheHandle
from switchback.decoder import EngineOptions, decode_speculative_sampled
from switchback.events import FakeClock
from switchback.oracle import (
    Tree,
    as_distribution,
    enumerate_algorithm,
    exact_fraction,
    target_sequence_distribution,
    total_variation_distance,
)
from switchback.types import ConfigError, DecodeConfig

VOCAB = 4
PROMPT = (1, 2)
MASKED_LOGIT = -1000.0

# Every distribution over four tokens that is uniform on a nonempty subset whose
# size divides the vocabulary. These are exactly the ones FP32 softmax
# represents without error.
SUPPORTS: list[tuple[int, ...]] = [
    support for size in (1, 2, 4) for support in itertools.combinations(range(VOCAB), size)
]


def logits_for(support: tuple[int, ...]) -> list[float]:
    return [0.0 if token in support else MASKED_LOGIT for token in range(VOCAB)]


def distribution_for(support: tuple[int, ...]) -> tuple[Fraction, ...]:
    share = Fraction(1, len(support))
    return tuple(share if token in support else Fraction(0) for token in range(VOCAB))


@dataclass
class ScriptedProbabilityAdapter:
    """A causal model whose row at each prefix is a chosen uniform subset."""

    supports: dict[tuple[int, ...], tuple[int, ...]]
    name: str = "scripted"
    vocab_size: int = VOCAB
    checks: bool = True
    widths: list[int] = field(default_factory=list)

    def support_for(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        return self.supports.get(prefix[-2:], (0, 1))

    def new_cache(self) -> CacheHandle:
        return CacheHandle(name=self.name, backend=None, tokens=[], checks=self.checks)

    def backend_length(self, cache: CacheHandle) -> int | None:
        return None

    def forward(self, ids: torch.Tensor, cache: CacheHandle) -> torch.Tensor:
        fed = [int(value) for value in ids[0].tolist()]
        self.widths.append(len(fed))
        prefix = tuple(cache.tokens)
        rows = torch.zeros((1, len(fed), self.vocab_size), dtype=torch.float32)
        for index, token in enumerate(fed):
            prefix = (*prefix, token)
            rows[0, index] = torch.tensor(logits_for(self.support_for(prefix)))
        cache.extend(fed)
        return rows

    def crop(self, cache: CacheHandle, length: int) -> None:
        cache.crop_to(length)

    def cache_length(self, cache: CacheHandle) -> int:
        return cache.length


def build_pair(seed: int) -> tuple[ScriptedProbabilityAdapter, ScriptedProbabilityAdapter]:
    """Deterministic target and draft models keyed on the last two tokens."""
    import random

    rng = random.Random(seed)
    keys = [(a, b) for a in range(VOCAB + 2) for b in range(VOCAB + 2)]
    target = {key: rng.choice(SUPPORTS) for key in keys}
    draft = {key: rng.choice(SUPPORTS) for key in keys}
    return (
        ScriptedProbabilityAdapter(target, "target"),
        ScriptedProbabilityAdapter(draft, "draft"),
    )


def reference_tree(adapter: ScriptedProbabilityAdapter, length: int) -> Tree:
    """The target's own law, built from the same rows the decoder will see."""
    rows: dict[tuple[int, ...], tuple[Fraction, ...]] = {}
    frontier = [PROMPT]
    for _ in range(length + 1):
        following: list[tuple[int, ...]] = []
        for prefix in frontier:
            rows[prefix] = as_distribution(distribution_for(adapter.support_for(prefix)))
            following.extend((*prefix, token) for token in range(VOCAB))
        frontier = following
    return Tree(vocab_size=VOCAB, rows=rows)


def constant_pair(
    target_support: tuple[int, ...], draft_support: tuple[int, ...]
) -> tuple[ScriptedProbabilityAdapter, ScriptedProbabilityAdapter]:
    """Adapters whose every row is a fixed uniform subset.

    Random trees are convenient but they do not guarantee an interesting
    residual: if ``max(p - q, 0)`` happens to be supported on a single token,
    drawing from it and taking its argmax are the same operation, and a test
    built on such a tree cannot detect a broken correction. These explicit pairs
    pin the cases that matter.
    """

    class Constant(ScriptedProbabilityAdapter):
        def support_for(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
            return self.fixed  # type: ignore[attr-defined]

    target = Constant({}, "target")
    target.fixed = target_support  # type: ignore[attr-defined]
    draft = Constant({}, "draft")
    draft.fixed = draft_support  # type: ignore[attr-defined]
    return target, draft


# Constructed cases, each chosen for the path it forces.
CONSTANT_CASES: list[tuple[tuple[int, ...], tuple[int, ...], str]] = [
    ((0, 1, 2, 3), (0,), "residual over three tokens"),
    ((0, 1, 2, 3), (0, 1), "residual over two tokens"),
    ((0, 1), (0, 1, 2, 3), "draft broader than the target"),
    ((0, 1), (2, 3), "disjoint supports, every candidate rejected"),
    ((0,), (0, 1), "target is deterministic, draft is not"),
    ((0, 1), (0, 1), "identical rows, every candidate accepted"),
]


def residual_support_sizes(
    target: ScriptedProbabilityAdapter,
    draft: ScriptedProbabilityAdapter,
    gamma: int,
    length: int,
) -> list[int]:
    """Support size of every residual actually drawn from during enumeration."""
    import switchback.sampling as sampling

    sizes: list[int] = []
    original = sampling.draw_correction

    def spy(target_probabilities, draft_probabilities, source):
        weights = sampling.positive_residual(target_probabilities, draft_probabilities)
        sizes.append(int((weights > 0).sum()))
        return original(target_probabilities, draft_probabilities, source)

    sampling.draw_correction = spy
    try:
        decoder_law(target, draft, gamma, length)
    finally:
        sampling.draw_correction = original
    return sizes


def sampled_config(length: int) -> DecodeConfig:
    return DecodeConfig(
        mode="sample",
        temperature=1.0,
        max_new_tokens=length,
        eos_policy="respect",
        seed=7,
    )


def decoder_law(
    target: ScriptedProbabilityAdapter,
    draft: ScriptedProbabilityAdapter,
    gamma: int,
    length: int,
) -> dict[object, Fraction]:
    def run(source: object) -> tuple[int, ...]:
        result = decode_speculative_sampled(
            target,
            draft,
            list(PROMPT),
            sampled_config(length),
            [],
            gamma=gamma,
            source=source,  # type: ignore[arg-type]
            clock=FakeClock(),
            options=EngineOptions(correctness_checks=True, device="cpu"),
        )
        return result.output_ids

    return enumerate_algorithm(run, max_paths=200_000)


# --- the milestone 5 gate --------------------------------------------------


def test_fp32_softmax_is_exact_for_these_rows() -> None:
    """The premise the exactness of every test below rests on."""
    from switchback.sampling import softmax_probabilities

    for support in SUPPORTS:
        probabilities = softmax_probabilities(
            torch.tensor(logits_for(support), dtype=torch.float32), 1.0
        )
        exact = tuple(exact_fraction(value) for value in probabilities)
        assert exact == distribution_for(support), support
        assert sum(exact, Fraction(0)) == 1


@pytest.mark.parametrize("case", range(6))
@pytest.mark.parametrize("gamma", [1, 2])
def test_sampled_speculation_reproduces_the_target_distribution(case: int, gamma: int) -> None:
    """Exact, through the cache, the softmax and the rollback."""
    target, draft = build_pair(500 + case)
    length = 3
    law = decoder_law(target, draft, gamma, length)
    expected = target_sequence_distribution(reference_tree(target, length), PROMPT, length)
    assert sum(law.values()) == 1
    assert total_variation_distance(law, expected) == 0, (case, gamma)


@pytest.mark.parametrize(
    ("target_support", "draft_support", "label"),
    CONSTANT_CASES,
    ids=[case[2] for case in CONSTANT_CASES],
)
@pytest.mark.parametrize("gamma", [1, 2, 3])
def test_constructed_cases_reproduce_the_target_distribution(
    target_support: tuple[int, ...],
    draft_support: tuple[int, ...],
    label: str,
    gamma: int,
) -> None:
    """The same gate on cases chosen for the path each one forces."""
    target, draft = constant_pair(target_support, draft_support)
    length = 3
    law = decoder_law(target, draft, gamma, length)
    expected = target_sequence_distribution(reference_tree(target, length), PROMPT, length)
    assert sum(law.values()) == 1
    assert total_variation_distance(law, expected) == 0, (label, gamma)


def test_the_gate_actually_exercises_multi_token_residuals() -> None:
    """Otherwise a correction bug could hide behind single-token residuals."""
    target, draft = constant_pair((0, 1, 2, 3), (0,))
    sizes = residual_support_sizes(target, draft, 2, 3)
    assert sizes
    assert max(sizes) >= 3, sizes


@pytest.mark.parametrize("case", range(4))
def test_the_block_length_does_not_change_the_law(case: int) -> None:
    """Two draft lengths must produce the same distribution, not merely a valid one."""
    target, draft = build_pair(700 + case)
    first = decoder_law(target, draft, 1, 3)
    second = decoder_law(target, draft, 2, 3)
    assert total_variation_distance(first, second) == 0


def test_a_broken_residual_is_detected_by_this_gate() -> None:
    """If the gate passes for a wrong sampler it proves nothing.

    Replacing the residual draw with an argmax is the specific shortcut SPEC.md
    section 4.1 forbids, and it must move the output distribution.
    """
    import switchback.sampling as sampling

    # A pair whose residual is supported on three tokens, so an argmax and a
    # draw are genuinely different operations.
    target, draft = constant_pair((0, 1, 2, 3), (0,))
    expected = target_sequence_distribution(reference_tree(target, 3), PROMPT, 3)
    original = sampling.draw_correction

    def argmax_correction(target_probabilities, draft_probabilities, source):
        weights = sampling.positive_residual(target_probabilities, draft_probabilities)
        return int(weights.argmax())

    sampling.draw_correction = argmax_correction
    try:
        law = decoder_law(target, draft, 2, 3)
    finally:
        sampling.draw_correction = original
    assert total_variation_distance(law, expected) > 0


def test_the_target_is_queried_once_per_block_with_the_full_proposal_list() -> None:
    target, draft = build_pair(11)
    decode_speculative_sampled(
        target,
        draft,
        list(PROMPT),
        sampled_config(6),
        [],
        gamma=2,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True, device="cpu"),
    )
    # Prefill of the prompt, then verification calls of width gamma + 1, and a
    # width-1 target-only step for a lone remaining token.
    assert target.widths[0] == len(PROMPT)
    assert set(target.widths[1:]) <= {1, 2, 3}
    assert 3 in target.widths[1:]


# --- configuration and reproducibility ------------------------------------


def test_greedy_configurations_are_refused() -> None:
    target, draft = build_pair(1)
    with pytest.raises(ConfigError, match="sample mode"):
        decode_speculative_sampled(
            target,
            draft,
            list(PROMPT),
            DecodeConfig("greedy", 1.0, 4, "respect", 1),
            [],
            gamma=2,
            clock=FakeClock(),
        )


def test_gamma_must_be_positive() -> None:
    target, draft = build_pair(1)
    with pytest.raises(ConfigError, match="gamma >= 1"):
        decode_speculative_sampled(
            target, draft, list(PROMPT), sampled_config(4), [], gamma=0, clock=FakeClock()
        )


def test_the_same_seed_reproduces_the_same_output() -> None:
    target, draft = build_pair(3)
    outputs = {
        decode_speculative_sampled(
            target,
            draft,
            list(PROMPT),
            sampled_config(12),
            [],
            gamma=4,
            clock=FakeClock(),
            options=EngineOptions(correctness_checks=True, device="cpu"),
        ).output_ids
        for _ in range(3)
    }
    assert len(outputs) == 1


def test_different_seeds_diverge() -> None:
    target, draft = build_pair(3)
    outputs = set()
    for seed in range(8):
        config = DecodeConfig("sample", 1.0, 12, "respect", seed)
        outputs.add(
            decode_speculative_sampled(
                target,
                draft,
                list(PROMPT),
                config,
                [],
                gamma=4,
                clock=FakeClock(),
                options=EngineOptions(correctness_checks=True, device="cpu"),
            ).output_ids
        )
    assert len(outputs) > 1


def test_requests_do_not_share_state() -> None:
    target, draft = build_pair(5)
    config = sampled_config(10)
    first = decode_speculative_sampled(
        target,
        draft,
        list(PROMPT),
        config,
        [],
        gamma=4,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True, device="cpu"),
    )
    decode_speculative_sampled(
        target,
        draft,
        [3, 3],
        config,
        [],
        gamma=4,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True, device="cpu"),
    )
    again = decode_speculative_sampled(
        target,
        draft,
        list(PROMPT),
        config,
        [],
        gamma=4,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True, device="cpu"),
    )
    assert first.output_ids == again.output_ids


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 9])
@pytest.mark.parametrize("gamma", [1, 2, 4])
def test_budget_is_met_exactly(budget: int, gamma: int) -> None:
    target, draft = build_pair(2)
    result = decode_speculative_sampled(
        target,
        draft,
        list(PROMPT),
        sampled_config(budget),
        [],
        gamma=gamma,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True, device="cpu"),
    )
    assert len(result.output_ids) == budget


def test_eos_terminates_and_truncates_the_rest_of_the_block() -> None:
    target, draft = build_pair(4)
    result = decode_speculative_sampled(
        target,
        draft,
        list(PROMPT),
        DecodeConfig("sample", 1.0, 12, "respect", 2),
        [3],
        gamma=4,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True, device="cpu"),
    )
    if result.termination == "eos":
        assert result.output_ids[-1] == 3
        assert 3 not in result.output_ids[:-1]


def test_suppressed_eos_is_never_emitted() -> None:
    target, draft = build_pair(4)
    result = decode_speculative_sampled(
        target,
        draft,
        list(PROMPT),
        DecodeConfig("sample", 1.0, 12, "suppress_until_budget", 2),
        [3],
        gamma=4,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True, device="cpu"),
    )
    assert 3 not in result.output_ids
    assert len(result.output_ids) == 12
