"""Generated checks on the validation boundary of the public configuration."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from switchback.types import MAX_CONTEXT_TOKENS, ConfigError, DecodeConfig

modes = st.sampled_from(["greedy", "sample"])
policies = st.sampled_from(["respect", "suppress_until_budget"])
budgets = st.integers(min_value=1, max_value=MAX_CONTEXT_TOKENS)
seeds = st.integers(min_value=0, max_value=2**31 - 1)
positive_temperatures = st.floats(
    min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False
)


@given(temperature=positive_temperatures, budget=budgets, policy=policies, seed=seeds)
def test_every_in_range_configuration_is_accepted(
    temperature: float, budget: int, policy: str, seed: int
) -> None:
    for mode in ("greedy", "sample"):
        config = DecodeConfig(
            mode=mode,  # type: ignore[arg-type]
            temperature=temperature,
            max_new_tokens=budget,
            eos_policy=policy,  # type: ignore[arg-type]
            seed=seed,
        )
        assert config.temperature == temperature
        assert config.max_new_tokens == budget


@given(
    budget=st.integers(min_value=-1000, max_value=0)
    | st.integers(min_value=MAX_CONTEXT_TOKENS + 1, max_value=MAX_CONTEXT_TOKENS + 10_000),
    mode=modes,
    policy=policies,
)
def test_out_of_range_budgets_are_always_rejected(budget: int, mode: str, policy: str) -> None:
    with pytest.raises(ConfigError):
        DecodeConfig(
            mode=mode,  # type: ignore[arg-type]
            temperature=1.0,
            max_new_tokens=budget,
            eos_policy=policy,  # type: ignore[arg-type]
            seed=0,
        )


@given(
    temperature=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    budget=budgets,
)
def test_sample_mode_never_accepts_a_non_positive_temperature(
    temperature: float, budget: int
) -> None:
    with pytest.raises(ConfigError):
        DecodeConfig(
            mode="sample",
            temperature=temperature,
            max_new_tokens=budget,
            eos_policy="respect",
            seed=0,
        )


@given(seed=st.integers(max_value=-1))
def test_negative_seeds_are_always_rejected(seed: int) -> None:
    with pytest.raises(ConfigError):
        DecodeConfig(
            mode="greedy",
            temperature=1.0,
            max_new_tokens=8,
            eos_policy="respect",
            seed=seed,
        )


@settings(max_examples=50)
@given(
    label=st.text(min_size=1, max_size=12).filter(lambda value: value not in ("greedy", "sample"))
)
def test_only_the_two_documented_modes_exist(label: str) -> None:
    with pytest.raises(ConfigError):
        DecodeConfig(
            mode=label,  # type: ignore[arg-type]
            temperature=1.0,
            max_new_tokens=8,
            eos_policy="respect",
            seed=0,
        )


@given(
    mode=modes,
    temperature=positive_temperatures,
    budget=budgets,
    policy=policies,
    seed=seeds,
)
def test_equal_fields_imply_equal_configurations(
    mode: str, temperature: float, budget: int, policy: str, seed: int
) -> None:
    # The config is the cache key for a reproducible request, so structural
    # equality has to hold for identical fields.
    kwargs = {
        "mode": mode,
        "temperature": temperature,
        "max_new_tokens": budget,
        "eos_policy": policy,
        "seed": seed,
    }
    assert DecodeConfig(**kwargs) == DecodeConfig(**kwargs)  # type: ignore[arg-type]
    assert hash(DecodeConfig(**kwargs)) == hash(DecodeConfig(**kwargs))  # type: ignore[arg-type]
