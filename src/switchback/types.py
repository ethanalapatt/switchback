"""Shared typed state for the Switchback engine.

Every public dataclass here is frozen. Configuration is explicit: nothing in this
module reads a model's generation defaults, an environment variable, or a global.

Tensor axis names used throughout the package:

- ``B`` batch, always 1 in the MVP.
- ``T`` positions fed to a forward call.
- ``V`` vocabulary size.
- ``S`` committed prompt-plus-output tokens for a request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

# Spec section 2: maximum prompt plus generation length.
MAX_CONTEXT_TOKENS = 4096

# Spec section 4.4: the controller's discrete action set. 0 means bypass.
CONTROLLER_ACTIONS: tuple[int, ...] = (0, 1, 2, 4, 8)

DecodeMode = Literal["greedy", "sample"]
EosPolicy = Literal["respect", "suppress_until_budget"]
TerminationReason = Literal["eos", "budget", "error"]


class ConfigError(ValueError):
    """A configuration value is missing, out of range, or internally inconsistent."""


class TokenizerCompatibilityError(ConfigError):
    """Draft and target tokenizers cannot be used for the same token stream."""


class NumericalSamplingError(RuntimeError):
    """A sampling distribution was nonfinite, unnormalizable, or had empty support.

    Spec section 4.1: raise rather than silently substituting another distribution.
    The run that produced this is invalid.
    """


@dataclass(frozen=True)
class DecodeConfig:
    """Complete generation contract for one request.

    Nothing is inherited from a model's ``generation_config``. Top-k, top-p,
    repetition penalty and friends are disabled by construction because this
    dataclass has no field for them (spec section 4.1).
    """

    mode: DecodeMode
    temperature: float
    max_new_tokens: int
    eos_policy: EosPolicy
    seed: int

    def __post_init__(self) -> None:
        if self.mode not in ("greedy", "sample"):
            raise ConfigError(f"mode must be 'greedy' or 'sample', got {self.mode!r}")
        if self.eos_policy not in ("respect", "suppress_until_budget"):
            raise ConfigError(f"unknown eos_policy {self.eos_policy!r}")
        if self.mode == "sample":
            if not (self.temperature > 0.0):
                raise ConfigError(
                    f"sample mode requires temperature > 0, got {self.temperature!r}"
                )
            if self.temperature != self.temperature or self.temperature == float("inf"):
                raise ConfigError(f"temperature must be finite, got {self.temperature!r}")
        if self.max_new_tokens < 1:
            raise ConfigError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.max_new_tokens > MAX_CONTEXT_TOKENS:
            raise ConfigError(
                f"max_new_tokens {self.max_new_tokens} exceeds the "
                f"{MAX_CONTEXT_TOKENS}-token context bound"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ConfigError(f"seed must be an int, got {self.seed!r}")
        if self.seed < 0:
            raise ConfigError(f"seed must be non-negative, got {self.seed}")


@dataclass(frozen=True)
class ModelSpec:
    """Immutable identity of one loaded checkpoint.

    ``revision`` must be a resolved commit SHA, never a branch name, so a run
    manifest names exactly the weights that were executed.
    """

    repo_id: str
    revision: str
    dtype: str
    device: str
    attn_implementation: str

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ConfigError("repo_id must not be empty")
        if self.revision in ("", "main", "master", "latest"):
            raise ConfigError(
                f"revision must be a resolved immutable commit, got {self.revision!r}"
            )


@dataclass(frozen=True)
class TokenizerCompatibility:
    """Result of comparing the draft and target tokenizers.

    Matching ``vocab_size`` in two configs does not establish compatibility
    (spec section 2), so every field below is checked independently.
    """

    compatible: bool
    target_vocab_size: int
    draft_vocab_size: int
    target_config_vocab_size: int
    draft_config_vocab_size: int
    checked_probe_strings: int
    failures: tuple[str, ...] = ()

    def raise_if_incompatible(self) -> None:
        if not self.compatible:
            joined = "; ".join(self.failures) or "unspecified mismatch"
            raise TokenizerCompatibilityError(f"draft/target tokenizer mismatch: {joined}")


@dataclass(frozen=True)
class BlockObservation:
    """What the decoder learned from one completed block.

    Timing fields are nanoseconds measured after GPU synchronization. They are
    supplied to the controller only; the sampler never sees them.
    """

    block_id: int
    action: int
    proposed: int
    accepted: int
    rejection_position: int | None
    draft_ns: int
    target_ns: int
    overhead_ns: int
    committed: int
    cache_length_after: int


@dataclass(frozen=True)
class DecodeResult:
    """Everything one request produced. Only committed tokens appear here."""

    run_id: str
    request_id: str
    output_ids: tuple[int, ...]
    token_release_ns: tuple[int, ...]
    termination: TerminationReason
    start_ns: int
    first_token_ns: int
    last_token_ns: int
    end_ns: int
    proposed: int
    accepted: int
    target_calls: int
    draft_calls: int
    controller_decisions: int
    bypass_decisions: int
    counters: dict[str, int | None] = field(default_factory=dict)


@runtime_checkable
class CacheHandle(Protocol):
    """Opaque handle to one model's KV cache plus the prefix it represents."""

    @property
    def length(self) -> int:
        """Number of token positions currently materialized in the cache."""


@runtime_checkable
class ModelAdapter(Protocol):
    """The only place Transformers cache/forward assumptions are allowed to live."""

    def new_cache(self) -> CacheHandle: ...

    def forward(self, ids: object, cache: CacheHandle) -> object:
        """ids ``[1, T]``; appends ``T`` positions; returns logits ``[1, T, V]``."""

    def crop(self, cache: CacheHandle, length: int) -> None: ...

    def cache_length(self, cache: CacheHandle) -> int: ...


@runtime_checkable
class Controller(Protocol):
    """Chooses the next draft length from information available before drafting."""

    def choose(self, state: object) -> int:
        """Return one of ``CONTROLLER_ACTIONS``."""

    def observe(self, block: BlockObservation) -> None: ...
