"""The single Qwen3 adapter.

This module owns every Hugging Face Transformers assumption in the project
(SPEC.md section 3). Milestone 1 covers loading, immutable revision resolution,
residency, and tokenizer compatibility. The pending-token forward/crop contract
arrives in milestone 3.

Tensor axis names: ``B`` batch (always 1), ``T`` positions fed to a forward
call, ``V`` vocabulary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from switchback.types import (
    MAX_CONTEXT_TOKENS,
    ConfigError,
    ModelSpec,
    TokenizerCompatibility,
)

# SPEC.md section 2: the required initial model pair.
TARGET_REPO = "Qwen/Qwen3-4B"
DRAFT_REPO = "Qwen/Qwen3-0.6B"

SUPPORTED_DTYPES: dict[str, str] = {
    "bfloat16": "bfloat16",
    "float32": "float32",
    "float16": "float16",
}

# Strings chosen to exercise byte-level BPE edges, not to look impressive:
# leading/trailing whitespace, tabs, CJK, emoji, digits, code punctuation, and
# the literal text of chat-template control tokens.
PROBE_STRINGS: tuple[str, ...] = (
    "",
    " ",
    "\n",
    "\t\t",
    "a",
    "Write a Python function for this task. Return code only.",
    "Solve the problem and state the final answer.",
    "def f(x: int) -> int:\n    return x * 2\n",
    "for i in range(10):\n\tprint(i)",
    "0123456789",
    "3.14159e-10",
    "  leading and trailing  ",
    "naïve café résumé",
    "日本語のテキスト",
    "中文测试字符串",
    "Здравствуй, мир",
    "emoji: 🙂🚀🧪",
    "<|im_start|>user",
    "<|im_end|>",
    "<|endoftext|>",
    "{'json': [1, 2, 3]}",
    "a" * 257,
    "​zero width",
    "mixed CASE and_snake-kebab",
)


class TokenizerLike(Protocol):
    """The tokenizer surface this project depends on."""

    def get_vocab(self) -> dict[str, int]: ...

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def pad_token_id(self) -> int | None: ...


@dataclass(frozen=True)
class LoadedModel:
    """One resident checkpoint plus the objects needed to drive it.

    ``logits_vocab_size`` is the output-embedding row count, which is the axis
    the sampler actually compares. It can differ from ``config.vocab_size`` and
    from the tokenizer's vocabulary; all three are recorded separately.
    """

    spec: ModelSpec
    model: Any
    tokenizer: Any
    config: Any
    logits_vocab_size: int
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]
    parameter_count: int


def resolve_revision(repo_id: str, revision: str = "main") -> str:
    """Resolve a branch name to an immutable commit SHA via the Hub API."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision)
    sha = info.sha
    if not isinstance(sha, str) or len(sha) != 40:
        raise ConfigError(f"could not resolve an immutable revision for {repo_id!r}")
    return sha


def _torch_dtype(name: str) -> Any:
    import torch

    if name not in SUPPORTED_DTYPES:
        raise ConfigError(
            f"dtype {name!r} is not supported; choose one of {sorted(SUPPORTED_DTYPES)}"
        )
    return getattr(torch, name)


def validate_spec(spec: ModelSpec) -> None:
    """Reject a model spec this build cannot honour, before any weights load."""
    import torch

    if spec.dtype not in SUPPORTED_DTYPES:
        raise ConfigError(
            f"dtype {spec.dtype!r} is not supported; choose one of {sorted(SUPPORTED_DTYPES)}"
        )
    device = spec.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ConfigError("device 'cuda' requested but torch reports no CUDA device")
        index = 0 if device == "cuda" else int(device.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise ConfigError(
                f"device {device!r} requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
    elif device != "cpu":
        raise ConfigError(f"device must be 'cpu' or 'cuda[:n]', got {device!r}")
    if spec.attn_implementation not in ("sdpa", "eager", "flash_attention_2"):
        raise ConfigError(f"unknown attn_implementation {spec.attn_implementation!r}")


def validate_context_window(repo_id: str, max_position_embeddings: int) -> None:
    """Refuse a checkpoint that cannot hold the bounded context from SPEC.md section 2."""
    if max_position_embeddings < MAX_CONTEXT_TOKENS:
        raise ConfigError(
            f"{repo_id} supports {max_position_embeddings} positions, below the "
            f"{MAX_CONTEXT_TOKENS}-token prompt-plus-generation bound"
        )


def assert_resident(model: Any, device: str) -> None:
    """Fail if any parameter or buffer lives somewhere other than ``device``.

    This is the "no CPU offload" gate from SPEC.md section 2. Accelerate's
    ``hf_device_map`` is checked too, because a dispatched model can look
    resident while silently streaming layers.
    """
    import torch

    expected = torch.device(device if device != "cuda" else "cuda:0")
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        offloaded = {
            name: where
            for name, where in device_map.items()
            if str(where) in {"cpu", "disk", "meta"}
        }
        if offloaded:
            raise ConfigError(f"model is offloaded, refusing to benchmark: {offloaded}")
    misplaced: list[str] = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        actual = tensor.device
        if actual.type != expected.type or (actual.index or 0) != (expected.index or 0):
            misplaced.append(f"{name}@{actual}")
        if len(misplaced) >= 5:
            break
    if misplaced:
        raise ConfigError(
            f"expected every tensor on {expected}, found {len(misplaced)}+ elsewhere: {misplaced}"
        )


def load_qwen(
    repo_id: str,
    revision: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
    attn_implementation: str = "sdpa",
    local_files_only: bool = False,
) -> LoadedModel:
    """Load one Qwen3 checkpoint fully resident on ``device``.

    No ``device_map``, no offload, no quantization, and no generation defaults
    are consulted. ``revision`` must already be an immutable commit SHA.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    spec = ModelSpec(
        repo_id=repo_id,
        revision=revision,
        dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
    )
    validate_spec(spec)
    config = AutoConfig.from_pretrained(
        repo_id, revision=revision, local_files_only=local_files_only
    )
    max_positions = int(getattr(config, "max_position_embeddings", 0))
    validate_context_window(repo_id, max_positions)
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, revision=revision, local_files_only=local_files_only
    )
    # ``Any`` because Transformers' ``PreTrainedModel.to`` overloads do not
    # describe the ``(device)`` form this project uses.
    model: Any = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        dtype=_torch_dtype(dtype),
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
        device_map=None,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    assert_resident(model, device)
    output_embeddings = model.get_output_embeddings()
    logits_vocab_size = int(output_embeddings.weight.shape[0])
    return LoadedModel(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        config=config,
        logits_vocab_size=logits_vocab_size,
        max_position_embeddings=max_positions,
        eos_token_ids=eos_token_ids(tokenizer, config),
        parameter_count=sum(p.numel() for p in model.parameters()),
    )


def eos_token_ids(tokenizer: Any, config: Any) -> tuple[int, ...]:
    """Every token ID that terminates a request, from the tokenizer and config.

    Qwen3 chat completions end with ``<|im_end|>``; the raw ``eos_token_id`` is
    ``<|endoftext|>``. Both are collected here so the EOS policy is explicit
    rather than inherited from a ``generation_config``.
    """
    found: list[int] = []
    raw = getattr(config, "eos_token_id", None)
    if isinstance(raw, int):
        found.append(raw)
    elif isinstance(raw, (list, tuple)):
        found.extend(int(value) for value in raw if isinstance(value, int))
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos, int):
        found.append(tokenizer_eos)
    vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for name in ("<|im_end|>", "<|endoftext|>"):
        if name in vocabulary:
            found.append(int(vocabulary[name]))
    return tuple(sorted(set(found)))


def check_tokenizer_compatibility(
    target_tokenizer: Any,
    draft_tokenizer: Any,
    target_config_vocab_size: int,
    draft_config_vocab_size: int,
    probes: Sequence[str] = PROBE_STRINGS,
) -> TokenizerCompatibility:
    """Compare two tokenizers thoroughly enough to share one token stream.

    SPEC.md section 2: equal ``vocab_size`` fields prove nothing. The full
    token-to-ID map, the added-token table, the special IDs, and the actual
    encodings of probe strings must all agree, because speculative decoding
    feeds draft-produced IDs straight into the target.
    """
    failures: list[str] = []
    target_vocab = target_tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    if target_vocab != draft_vocab:
        only_target = set(target_vocab) - set(draft_vocab)
        only_draft = set(draft_vocab) - set(target_vocab)
        disagreeing = [
            token
            for token in set(target_vocab) & set(draft_vocab)
            if target_vocab[token] != draft_vocab[token]
        ]
        failures.append(
            f"token-to-ID maps differ: {len(only_target)} target-only, "
            f"{len(only_draft)} draft-only, {len(disagreeing)} conflicting IDs "
            f"(examples: {sorted(disagreeing)[:3] or sorted(only_target | only_draft)[:3]})"
        )
    if target_config_vocab_size != draft_config_vocab_size:
        failures.append(
            f"config vocab_size differs: target {target_config_vocab_size} "
            f"vs draft {draft_config_vocab_size}"
        )
    for name in ("eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"):
        left = getattr(target_tokenizer, name, None)
        right = getattr(draft_tokenizer, name, None)
        if left != right:
            failures.append(f"{name} differs: target {left!r} vs draft {right!r}")
    left_added = _added_tokens(target_tokenizer)
    right_added = _added_tokens(draft_tokenizer)
    if left_added != right_added:
        failures.append(
            f"added-token tables differ: {len(left_added)} target entries "
            f"vs {len(right_added)} draft entries"
        )
    mismatched_probes: list[str] = []
    for probe in probes:
        left_ids = target_tokenizer.encode(probe, add_special_tokens=False)
        right_ids = draft_tokenizer.encode(probe, add_special_tokens=False)
        if list(left_ids) != list(right_ids):
            mismatched_probes.append(probe)
    if mismatched_probes:
        failures.append(
            f"{len(mismatched_probes)}/{len(probes)} probe strings encode differently "
            f"(first: {mismatched_probes[0]!r})"
        )
    return TokenizerCompatibility(
        compatible=not failures,
        target_vocab_size=len(target_vocab),
        draft_vocab_size=len(draft_vocab),
        target_config_vocab_size=target_config_vocab_size,
        draft_config_vocab_size=draft_config_vocab_size,
        checked_probe_strings=len(probes),
        failures=tuple(failures),
    )


def _added_tokens(tokenizer: Any) -> dict[int, str]:
    table = getattr(tokenizer, "added_tokens_decoder", None)
    if not table:
        return {}
    return {int(index): str(token) for index, token in table.items()}


def check_logits_vocab_match(target: LoadedModel, draft: LoadedModel) -> None:
    """Both models must emit logits over the same axis or ``p/q`` is meaningless."""
    if target.logits_vocab_size != draft.logits_vocab_size:
        raise ConfigError(
            f"logits vocabulary mismatch: target {target.logits_vocab_size} "
            f"vs draft {draft.logits_vocab_size}; the acceptance ratio p/q is undefined"
        )


def render_chat_prompt(tokenizer: Any, instruction: str) -> list[int]:
    """Token IDs for one user instruction under the target's chat template.

    ``enable_thinking=False`` and ``add_generation_prompt=True`` are explicit
    (SPEC.md section 2). Every engine receives these identical prompt IDs.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(value) for value in ids]
