"""Tokenizer parity is the precondition for feeding draft IDs to the target.

These use hand-built tokenizers so every failure mode is reachable offline; the
real Qwen pair is checked in tests/integration under the download marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from switchback.models.qwen import (
    PROBE_STRINGS,
    check_logits_vocab_match,
    check_tokenizer_compatibility,
    eos_token_ids,
    validate_context_window,
)
from switchback.types import MAX_CONTEXT_TOKENS, ConfigError


@dataclass
class FakeTokenizer:
    """Minimal tokenizer over single characters, with configurable defects."""

    vocabulary: dict[str, int]
    eos_token_id: int | None = 2
    bos_token_id: int | None = 1
    pad_token_id: int | None = 0
    unk_token_id: int | None = 3
    added_tokens_decoder: dict[int, str] = field(default_factory=dict)
    shift_encoding_by: int = 0

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.vocabulary.get(character, self.unk_token_id or 3) for character in text]
        return [value + self.shift_encoding_by for value in ids]


def standard_vocabulary() -> dict[str, int]:
    table = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for offset, character in enumerate(sorted(set("".join(PROBE_STRINGS)))):
        table[character] = 4 + offset
    return table


def build_pair() -> tuple[FakeTokenizer, FakeTokenizer]:
    vocabulary = standard_vocabulary()
    return FakeTokenizer(vocabulary), FakeTokenizer(dict(vocabulary))


def test_identical_tokenizers_are_compatible() -> None:
    target, draft = build_pair()
    report = check_tokenizer_compatibility(target, draft, 4096, 4096)
    assert report.compatible, report.failures
    assert report.failures == ()
    assert report.checked_probe_strings == len(PROBE_STRINGS)
    assert report.target_vocab_size == report.draft_vocab_size


def test_extra_draft_token_is_detected() -> None:
    target, draft = build_pair()
    draft.vocabulary["<|extra|>"] = max(draft.vocabulary.values()) + 1
    report = check_tokenizer_compatibility(target, draft, 4096, 4096)
    assert not report.compatible
    assert any("token-to-ID maps differ" in failure for failure in report.failures)


def test_conflicting_id_for_a_shared_token_is_detected() -> None:
    target, draft = build_pair()
    # Same surface form, different ID: the most dangerous case, because the
    # vocabularies are the same size and the failure only shows up in output.
    draft.vocabulary["a"], draft.vocabulary["b"] = (
        draft.vocabulary["b"],
        draft.vocabulary["a"],
    )
    report = check_tokenizer_compatibility(target, draft, 4096, 4096)
    assert not report.compatible
    assert report.target_vocab_size == report.draft_vocab_size


def test_matching_vocab_size_does_not_imply_compatibility() -> None:
    target, draft = build_pair()
    draft.shift_encoding_by = 1
    report = check_tokenizer_compatibility(target, draft, 151936, 151936)
    assert report.target_config_vocab_size == report.draft_config_vocab_size
    assert not report.compatible
    assert any("probe strings encode differently" in failure for failure in report.failures)


def test_config_vocab_size_mismatch_is_reported() -> None:
    target, draft = build_pair()
    report = check_tokenizer_compatibility(target, draft, 151936, 151643)
    assert not report.compatible
    assert any("config vocab_size differs" in failure for failure in report.failures)


def test_special_token_mismatch_is_reported() -> None:
    target, draft = build_pair()
    draft.eos_token_id = 99
    report = check_tokenizer_compatibility(target, draft, 4096, 4096)
    assert not report.compatible
    assert any("eos_token_id differs" in failure for failure in report.failures)


def test_added_token_table_mismatch_is_reported() -> None:
    target, draft = build_pair()
    target.added_tokens_decoder = {2: "<eos>"}
    report = check_tokenizer_compatibility(target, draft, 4096, 4096)
    assert not report.compatible
    assert any("added-token tables differ" in failure for failure in report.failures)


def test_every_failure_is_collected_not_just_the_first() -> None:
    target, draft = build_pair()
    draft.eos_token_id = 99
    draft.bos_token_id = 98
    report = check_tokenizer_compatibility(target, draft, 151936, 151643)
    assert len(report.failures) >= 3


class _Loaded:
    def __init__(self, size: int) -> None:
        self.logits_vocab_size = size


def test_logits_vocabulary_mismatch_is_fatal() -> None:
    with pytest.raises(ConfigError, match="p/q is undefined"):
        check_logits_vocab_match(_Loaded(151936), _Loaded(151643))  # type: ignore[arg-type]
    check_logits_vocab_match(_Loaded(151936), _Loaded(151936))  # type: ignore[arg-type]


def test_context_bound_rejects_a_short_checkpoint() -> None:
    with pytest.raises(ConfigError, match="below the"):
        validate_context_window("tiny/model", MAX_CONTEXT_TOKENS - 1)
    validate_context_window("tiny/model", MAX_CONTEXT_TOKENS)


class _Config:
    def __init__(self, eos: object) -> None:
        self.eos_token_id = eos


def test_eos_ids_merge_config_tokenizer_and_chat_end_token() -> None:
    target, _ = build_pair()
    target.vocabulary["<|im_end|>"] = 151645
    target.eos_token_id = 151643
    ids = eos_token_ids(target, _Config([151645, 151643]))
    assert ids == (151643, 151645)


def test_eos_ids_accept_a_scalar_config_value() -> None:
    target, _ = build_pair()
    target.eos_token_id = None
    assert eos_token_ids(target, _Config(7)) == (7,)
