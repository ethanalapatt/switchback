"""Baseline engines must never inherit a hidden generation default.

Each check here corresponds to a way Transformers reintroduces a default that
would change the sampled distribution or the output length.
"""

from __future__ import annotations

import pytest

from switchback.models.hf_baselines import (
    DYNAMIC_CONFIDENCE_THRESHOLD,
    DYNAMIC_MAX_DRAFT_TOKENS,
    DYNAMIC_SCHEDULE,
    explicit_generation_config,
)
from switchback.types import DecodeConfig

EOS = (151643, 151645)


def config(**overrides: object) -> DecodeConfig:
    base: dict[str, object] = {
        "mode": "sample",
        "temperature": 0.7,
        "max_new_tokens": 128,
        "eos_policy": "respect",
        "seed": 11,
    }
    base.update(overrides)
    return DecodeConfig(**base)  # type: ignore[arg-type]


def test_truncation_warpers_are_disabled() -> None:
    # GenerationConfig() defaults to top_k=50; Qwen3 ships top_k/top_p/temperature
    # in its own generation_config.json. MVP samples from the full distribution.
    generation = explicit_generation_config(config(), EOS, 151643)
    assert generation.top_k == 0
    assert generation.top_p == 1.0
    assert generation.min_p is None
    assert generation.typical_p == 1.0
    assert generation.epsilon_cutoff == 0.0
    assert generation.eta_cutoff == 0.0


def test_penalties_and_beams_are_neutral() -> None:
    generation = explicit_generation_config(config(), EOS, 151643)
    assert generation.repetition_penalty == 1.0
    assert generation.encoder_repetition_penalty == 1.0
    assert generation.no_repeat_ngram_size == 0
    assert generation.num_beams == 1
    assert generation.num_return_sequences == 1


def test_sampling_is_enabled_only_in_sample_mode() -> None:
    assert explicit_generation_config(config(), EOS, 151643).do_sample is True
    greedy = explicit_generation_config(config(mode="greedy", temperature=1.0), EOS, 151643)
    assert greedy.do_sample is False
    # Temperature must not leak into greedy decoding as a scaling factor.
    assert greedy.temperature == 1.0


def test_temperature_is_carried_through_in_sample_mode() -> None:
    generation = explicit_generation_config(config(temperature=0.7), EOS, 151643)
    assert generation.temperature == pytest.approx(0.7)


def test_respect_policy_keeps_every_stop_token() -> None:
    generation = explicit_generation_config(config(), EOS, 151643)
    assert generation.eos_token_id == list(EOS)


def test_suppress_policy_removes_stop_tokens_without_a_minimum_length() -> None:
    generation = explicit_generation_config(config(eos_policy="suppress_until_budget"), EOS, 151643)
    assert generation.eos_token_id is None
    # A non-None min_new_tokens makes Transformers derive min_length from the
    # prompt and install a processor that assisted generation refuses.
    assert generation.min_new_tokens is None
    assert generation.min_length == 0
    assert generation.max_new_tokens == 128


def test_cache_is_enabled_and_scores_are_not_retained() -> None:
    generation = explicit_generation_config(config(), EOS, 151643)
    assert generation.use_cache is True
    assert generation.output_scores is False
    assert generation.output_logits is False
    assert generation.return_dict_in_generate is False


def test_dynamic_baseline_options_match_the_locked_specification() -> None:
    assert DYNAMIC_CONFIDENCE_THRESHOLD == 0.4
    assert DYNAMIC_MAX_DRAFT_TOKENS == 8
    assert DYNAMIC_SCHEDULE == "constant"
