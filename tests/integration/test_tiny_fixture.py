"""Offline checks on the tiny fixture: no weights, no GPU, no network."""

from __future__ import annotations

import torch

from switchback.models.qwen import check_tokenizer_compatibility
from switchback.models.tiny import (
    EOS_ID,
    TINY_VOCAB_SIZE,
    TinyTokenizer,
    build_tiny_model,
    tiny_config,
    tiny_pair,
)


def test_same_seed_reproduces_identical_logits() -> None:
    ids = torch.tensor([[4, 20, 36, 7, 50, 9]])
    outputs = []
    for _ in range(2):
        model = build_tiny_model(seed=1234)
        with torch.inference_mode():
            outputs.append(model.model(input_ids=ids).logits)
    assert torch.equal(outputs[0], outputs[1])


def test_different_seeds_produce_different_models() -> None:
    ids = torch.tensor([[4, 20, 36]])
    with torch.inference_mode():
        left = build_tiny_model(seed=1).model(input_ids=ids).logits
        right = build_tiny_model(seed=2).model(input_ids=ids).logits
    assert not torch.equal(left, right)


def test_logits_have_the_documented_axes() -> None:
    model = build_tiny_model(seed=1234)
    ids = torch.tensor([[4, 5, 6, 7, 8]])
    with torch.inference_mode():
        logits = model.model(input_ids=ids).logits
    assert logits.shape == (1, ids.shape[1], TINY_VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_distribution_is_not_degenerate() -> None:
    # A near-uniform softmax would make acceptance tests pass for free, and a
    # one-hot softmax would hide residual-distribution bugs.
    model = build_tiny_model(seed=1234)
    ids = torch.tensor([[4, 20, 36, 7, 50, 9, 11, 30]])
    with torch.inference_mode():
        probabilities = model.model(input_ids=ids).logits[0].softmax(-1)
    top = probabilities.max(-1).values
    assert float(top.min()) > 1.5 / TINY_VOCAB_SIZE
    assert float(top.max()) < 0.999


def test_greedy_argmax_is_deterministic_across_rebuilds() -> None:
    ids = torch.tensor([[4, 20, 36, 7, 50, 9]])
    sequences = []
    for _ in range(2):
        model = build_tiny_model(seed=99)
        current = ids.clone()
        for _ in range(8):
            with torch.inference_mode():
                logits = model.model(input_ids=current).logits[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            current = torch.cat([current, nxt], dim=1)
        sequences.append(current)
    assert torch.equal(sequences[0], sequences[1])


def test_fixture_revision_encodes_the_build_arguments() -> None:
    a = build_tiny_model(seed=1, hidden_size=64, layers=2)
    b = build_tiny_model(seed=1, hidden_size=32, layers=2)
    assert a.spec.revision != b.spec.revision
    assert a.spec.revision.startswith("fixture-")


def test_tiny_pair_shares_a_vocabulary() -> None:
    target, draft = tiny_pair()
    assert target.logits_vocab_size == draft.logits_vocab_size
    assert target.parameter_count > draft.parameter_count
    report = check_tokenizer_compatibility(
        target.tokenizer,
        draft.tokenizer,
        target.config.vocab_size,
        draft.config.vocab_size,
    )
    assert report.compatible, report.failures


def test_tokenizer_round_trips_printable_text() -> None:
    tokenizer = TinyTokenizer()
    text = "def f(x): return x + 1"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode(text, add_special_tokens=True)[0] == tokenizer.bos_token_id
    assert tokenizer.eos_token_id == EOS_ID


def test_config_supports_the_full_context_bound() -> None:
    assert tiny_config().max_position_embeddings >= 4096
