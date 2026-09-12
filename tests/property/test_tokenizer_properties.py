"""Generated checks on the tokenizer compatibility contract."""

from __future__ import annotations

from dataclasses import dataclass, field

from hypothesis import given, settings
from hypothesis import strategies as st

from switchback.models.qwen import check_tokenizer_compatibility

SURFACES = ["<pad>", "<bos>", "<eos>", "<unk>", "a", "b", "c", " ", "\n", "1", "2"]
PROBES = ("a", "b", "abc", " a\n", "", "12")


@dataclass
class ListTokenizer:
    """Tokenizer whose vocabulary is an explicit ordering of ``SURFACES``."""

    order: list[str]
    eos_token_id: int | None = 2
    bos_token_id: int | None = 1
    pad_token_id: int | None = 0
    unk_token_id: int | None = 3
    added_tokens_decoder: dict[int, str] = field(default_factory=dict)

    def get_vocab(self) -> dict[str, int]:
        return {surface: index for index, surface in enumerate(self.order)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        table = self.get_vocab()
        return [table.get(character, table["<unk>"]) for character in text]


orderings = st.permutations(SURFACES)


@settings(max_examples=100)
@given(order=orderings)
def test_a_tokenizer_is_always_compatible_with_itself(order: list[str]) -> None:
    tokenizer = ListTokenizer(list(order))
    report = check_tokenizer_compatibility(tokenizer, tokenizer, 32, 32, probes=PROBES)
    assert report.compatible
    assert report.failures == ()


@settings(max_examples=100)
@given(order=orderings)
def test_compatibility_is_symmetric(order: list[str]) -> None:
    left = ListTokenizer(list(order))
    right = ListTokenizer(list(SURFACES))
    forward = check_tokenizer_compatibility(left, right, 32, 32, probes=PROBES)
    backward = check_tokenizer_compatibility(right, left, 32, 32, probes=PROBES)
    assert forward.compatible == backward.compatible


@settings(max_examples=100)
@given(order=orderings)
def test_any_reordering_of_ids_is_detected(order: list[str]) -> None:
    reference = ListTokenizer(list(SURFACES))
    candidate = ListTokenizer(list(order))
    report = check_tokenizer_compatibility(reference, candidate, 32, 32, probes=PROBES)
    assert report.compatible == (list(order) == SURFACES)


@settings(max_examples=100)
@given(
    order=orderings,
    target_size=st.integers(min_value=1, max_value=200_000),
    draft_size=st.integers(min_value=1, max_value=200_000),
)
def test_the_verdict_is_exactly_the_absence_of_failures(
    order: list[str], target_size: int, draft_size: int
) -> None:
    report = check_tokenizer_compatibility(
        ListTokenizer(list(SURFACES)),
        ListTokenizer(list(order)),
        target_size,
        draft_size,
        probes=PROBES,
    )
    assert report.compatible == (report.failures == ())
    assert report.checked_probe_strings == len(PROBES)


@settings(max_examples=50)
@given(size=st.integers(min_value=1, max_value=200_000))
def test_identical_maps_with_different_config_sizes_are_still_rejected(
    size: int,
) -> None:
    tokenizer = ListTokenizer(list(SURFACES))
    report = check_tokenizer_compatibility(tokenizer, tokenizer, size, size + 1, probes=PROBES)
    assert not report.compatible
