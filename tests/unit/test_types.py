"""DecodeConfig and ModelSpec reject configurations the engine cannot honour."""

from __future__ import annotations

import dataclasses

import pytest

from switchback.types import (
    CONTROLLER_ACTIONS,
    MAX_CONTEXT_TOKENS,
    ConfigError,
    DecodeConfig,
    ModelSpec,
    TokenizerCompatibility,
    TokenizerCompatibilityError,
)


def greedy(**overrides: object) -> DecodeConfig:
    base: dict[str, object] = {
        "mode": "greedy",
        "temperature": 1.0,
        "max_new_tokens": 32,
        "eos_policy": "respect",
        "seed": 42,
    }
    base.update(overrides)
    return DecodeConfig(**base)  # type: ignore[arg-type]


def test_greedy_config_is_accepted() -> None:
    config = greedy()
    assert config.mode == "greedy"
    assert config.max_new_tokens == 32


def test_sample_mode_requires_positive_temperature() -> None:
    with pytest.raises(ConfigError, match="temperature"):
        greedy(mode="sample", temperature=0.0)
    with pytest.raises(ConfigError, match="temperature"):
        greedy(mode="sample", temperature=-0.5)


def test_greedy_mode_ignores_temperature() -> None:
    # Greedy is a separate deterministic algorithm; temperature is unused, so a
    # zero value must not be treated as an error.
    assert greedy(temperature=0.0).temperature == 0.0


def test_unknown_mode_and_eos_policy_are_rejected() -> None:
    with pytest.raises(ConfigError, match="mode"):
        greedy(mode="beam")
    with pytest.raises(ConfigError, match="eos_policy"):
        greedy(eos_policy="ignore")


def test_budget_must_be_positive_and_within_the_context_bound() -> None:
    with pytest.raises(ConfigError, match="max_new_tokens"):
        greedy(max_new_tokens=0)
    with pytest.raises(ConfigError, match="context bound"):
        greedy(max_new_tokens=MAX_CONTEXT_TOKENS + 1)
    assert greedy(max_new_tokens=MAX_CONTEXT_TOKENS).max_new_tokens == MAX_CONTEXT_TOKENS


def test_seed_must_be_a_non_negative_int() -> None:
    with pytest.raises(ConfigError, match="seed"):
        greedy(seed=-1)
    with pytest.raises(ConfigError, match="seed"):
        greedy(seed=True)
    with pytest.raises(ConfigError, match="seed"):
        greedy(seed=1.5)


def test_config_is_frozen() -> None:
    config = greedy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_new_tokens = 64  # type: ignore[misc]


def test_model_spec_refuses_a_mutable_revision() -> None:
    for revision in ("", "main", "master", "latest"):
        with pytest.raises(ConfigError, match="revision"):
            ModelSpec("Qwen/Qwen3-4B", revision, "bfloat16", "cuda", "sdpa")
    spec = ModelSpec("Qwen/Qwen3-4B", "a" * 40, "bfloat16", "cuda", "sdpa")
    assert spec.revision == "a" * 40


def test_model_spec_requires_a_repo_id() -> None:
    with pytest.raises(ConfigError, match="repo_id"):
        ModelSpec("", "a" * 40, "bfloat16", "cuda", "sdpa")


def test_incompatible_tokenizers_raise_with_the_reasons_attached() -> None:
    report = TokenizerCompatibility(
        compatible=False,
        target_vocab_size=10,
        draft_vocab_size=9,
        target_config_vocab_size=10,
        draft_config_vocab_size=9,
        checked_probe_strings=3,
        failures=("token-to-ID maps differ",),
    )
    with pytest.raises(TokenizerCompatibilityError, match="token-to-ID maps differ"):
        report.raise_if_incompatible()


def test_compatible_report_does_not_raise() -> None:
    TokenizerCompatibility(True, 10, 10, 10, 10, 3).raise_if_incompatible()


def test_bypass_is_part_of_the_action_set() -> None:
    assert CONTROLLER_ACTIONS[0] == 0
    assert sorted(CONTROLLER_ACTIONS) == list(CONTROLLER_ACTIONS)
