"""Deterministic tiny transformer fixture for offline CPU tests.

No weights are downloaded: the architecture is the real ``Qwen3ForCausalLM``
class from Transformers, instantiated from a small config with seeded random
parameters. That keeps the cache and attention code paths identical to the
benchmark models while running in milliseconds on a laptop or CI worker.

Tensor axis names: ``B`` batch, ``T`` positions, ``V`` vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from switchback.provenance import sha256_bytes
from switchback.types import ConfigError, ModelSpec

# A character-level vocabulary keeps encodings inspectable by eye in a failing
# test. Reserved IDs come first so EOS handling matches the real adapter shape.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
FIRST_CHARACTER_ID = 4
CHARACTERS = "".join(chr(code) for code in range(32, 127))
TINY_VOCAB_SIZE = FIRST_CHARACTER_ID + len(CHARACTERS)


@dataclass(frozen=True)
class TinyTokenizer:
    """Character-level tokenizer with a fixed, fully enumerable vocabulary."""

    name: str = "switchback-tiny"

    def get_vocab(self) -> dict[str, int]:
        vocabulary = {"<pad>": PAD_ID, "<bos>": BOS_ID, "<eos>": EOS_ID, "<unk>": UNK_ID}
        for offset, character in enumerate(CHARACTERS):
            vocabulary[character] = FIRST_CHARACTER_ID + offset
        return vocabulary

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        table = self.get_vocab()
        ids = [table.get(character, UNK_ID) for character in text]
        return [BOS_ID, *ids] if add_special_tokens else ids

    def decode(self, ids: list[int]) -> str:
        reverse = {index: token for token, index in self.get_vocab().items()}
        return "".join(reverse.get(int(index), "<unk>") for index in ids)

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    @property
    def bos_token_id(self) -> int:
        return BOS_ID

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def unk_token_id(self) -> int:
        return UNK_ID

    @property
    def added_tokens_decoder(self) -> dict[int, str]:
        return {PAD_ID: "<pad>", BOS_ID: "<bos>", EOS_ID: "<eos>", UNK_ID: "<unk>"}


@dataclass(frozen=True)
class TinyModel:
    """A seeded tiny causal LM plus the identity needed to record it in a manifest."""

    spec: ModelSpec
    model: Any
    tokenizer: TinyTokenizer
    config: Any
    logits_vocab_size: int
    parameter_count: int


def tiny_config(
    hidden_size: int = 64,
    layers: int = 2,
    heads: int = 4,
    kv_heads: int = 2,
    intermediate_size: int = 128,
    max_position_embeddings: int = 4096,
) -> Any:
    """Build a ``Qwen3Config`` small enough for exhaustive CPU testing."""
    from transformers import Qwen3Config

    if hidden_size % heads:
        raise ConfigError(f"hidden_size {hidden_size} must divide by heads {heads}")
    if heads % kv_heads:
        raise ConfigError(f"heads {heads} must be a multiple of kv_heads {kv_heads}")
    return Qwen3Config(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=hidden_size // heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        attention_dropout=0.0,
    )


def build_tiny_model(
    seed: int,
    hidden_size: int = 64,
    layers: int = 2,
    dtype: str = "float32",
    device: str = "cpu",
    attn_implementation: str = "eager",
    init_std: float = 0.5,
) -> TinyModel:
    """Instantiate a tiny causal LM with parameters drawn from ``seed``.

    Every parameter is overwritten from one seeded CPU generator, walked in
    sorted-name order, so the same arguments reproduce bitwise identical weights
    on the same torch build and no checkpoint has to be stored. Non-parameter
    buffers such as the rotary ``inv_freq`` table keep the values Transformers
    computed, which is why the model is built on a real device rather than on
    ``meta``. ``init_std`` is large relative to a trained model so the resulting
    logits are not numerically degenerate; a near-uniform softmax would make
    every acceptance test pass trivially.
    """
    import torch
    from transformers import Qwen3ForCausalLM

    config = tiny_config(hidden_size=hidden_size, layers=layers)
    torch.manual_seed(seed)
    # ``Any``: see the note in models/qwen.py about ``PreTrainedModel.to``.
    model: Any = Qwen3ForCausalLM(config)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            sample = torch.empty(parameter.shape, dtype=torch.float32)
            if name.endswith("norm.weight"):
                # RMSNorm gains start at one; random gains scramble scale badly.
                sample.fill_(1.0)
            else:
                sample.normal_(mean=0.0, std=init_std, generator=generator)
            parameter.copy_(sample.to(parameter.dtype))
    model = model.to(device=device, dtype=getattr(torch, dtype))
    model.eval()
    model.requires_grad_(False)
    identity = sha256_bytes(
        f"tiny/{seed}/{hidden_size}/{layers}/{dtype}/{init_std}/{TINY_VOCAB_SIZE}".encode()
    )[:16]
    return TinyModel(
        spec=ModelSpec(
            repo_id="switchback/tiny-qwen3",
            revision=f"fixture-{identity}",
            dtype=dtype,
            device=device,
            attn_implementation=attn_implementation,
        ),
        model=model,
        tokenizer=TinyTokenizer(),
        config=config,
        logits_vocab_size=TINY_VOCAB_SIZE,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )


def tiny_pair(target_seed: int = 1234, draft_seed: int = 5678) -> tuple[TinyModel, TinyModel]:
    """A target/draft fixture pair that shares a vocabulary but not weights."""
    target = build_tiny_model(seed=target_seed, hidden_size=64, layers=2)
    draft = build_tiny_model(seed=draft_seed, hidden_size=32, layers=1)
    return target, draft
