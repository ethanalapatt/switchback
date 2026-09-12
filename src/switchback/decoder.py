"""Request loop and commit logic.

Owns request state and termination. Emits only committed tokens (invariant I1)
and maintains the pending-token convention at every block boundary
(invariant I2). Milestone 3 implements the target-only engines; speculation is
added on top of this loop in milestone 4.

Tensor axes: ``ids`` ``[1, T]``, logits ``[1, T, V]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from switchback.cache import CacheHandle
from switchback.events import BlockEvent, Clock, EventSink, MonotonicClock, NullSink
from switchback.models.qwen import QwenAdapter
from switchback.sampling import (
    RandomSource,
    TorchRandomSource,
    forbidden_token_ids,
    greedy_token,
    softmax_probabilities,
)
from switchback.sampling import (
    target_only_step as _sample_from_row,
)
from switchback.types import (
    MAX_CONTEXT_TOKENS,
    ConfigError,
    DecodeConfig,
    DecodeResult,
)


@dataclass(frozen=True)
class EngineOptions:
    """Knobs that must be identical across every engine in one comparison.

    ``correctness_checks`` turns on the cache ledger assertions. They are cheap
    but not free, so the value is recorded with each run: enabling them for one
    engine and not another would make a latency comparison meaningless
    (CLAUDE.md).
    """

    correctness_checks: bool = True
    device: str = "cpu"


@dataclass
class RequestState:
    """Everything one in-flight request owns. Nothing is shared between requests."""

    request_id: str
    run_id: str
    prompt_ids: tuple[int, ...]
    committed: list[int] = field(default_factory=list)
    release_ns: list[int] = field(default_factory=list)
    target_calls: int = 0
    draft_calls: int = 0
    blocks: int = 0

    @property
    def sequence(self) -> list[int]:
        """``S``: every committed prompt-and-output token."""
        return [*self.prompt_ids, *self.committed]


def select_token(
    row: Any,
    config: DecodeConfig,
    forbidden: Sequence[int],
    source: RandomSource | None,
) -> int:
    """Pick one token from a logits row ``[V]`` under the decode configuration.

    Greedy and sampled decoding are separate algorithms, not one algorithm with
    a temperature of zero: greedy uses explicit smallest-ID tie breaking, and
    never consumes randomness.
    """
    if config.mode == "greedy":
        return greedy_token(row, forbidden_ids=forbidden)
    if source is None:
        raise ConfigError("sample mode requires a random source")
    probabilities = softmax_probabilities(row, config.temperature, forbidden_ids=forbidden)
    return _sample_from_row((), lambda _prefix: probabilities, source)


def decode_target_only(
    adapter: QwenAdapter,
    prompt_ids: Sequence[int],
    config: DecodeConfig,
    eos_token_ids: Sequence[int],
    request_id: str = "request",
    run_id: str = "run",
    source: RandomSource | None = None,
    sink: EventSink | None = None,
    clock: Clock | None = None,
    options: EngineOptions | None = None,
) -> DecodeResult:
    """``native_ar``: Switchback's own target-only decoding loop.

    Initialization follows SPEC.md section 4.3: prefill the target with the whole
    prompt ``P``, take the first output token ``z`` from the last prefill row,
    and set ``S = P + [z]``. The cache then represents ``P``, which is
    ``S[:-1]`` -- the pending-token convention, established before any
    speculation exists to complicate it.

    The first token counts toward the output budget and toward TTFT.
    """
    import torch

    options = options or EngineOptions()
    sink = sink if sink is not None else NullSink()
    clock = clock if clock is not None else MonotonicClock(device=options.device)
    if source is None and config.mode == "sample":
        source = TorchRandomSource(request_seed=config.seed, device=options.device)

    prompt = tuple(int(value) for value in prompt_ids)
    _validate_request(prompt, config)
    forbidden = forbidden_token_ids(config, eos_token_ids)
    stop_ids = set() if forbidden else {int(value) for value in eos_token_ids}

    state = RequestState(request_id=request_id, run_id=run_id, prompt_ids=prompt)
    adapter.checks = options.correctness_checks
    cache: CacheHandle = adapter.new_cache()
    device = options.device

    start_ns = clock.mark()
    block_start = start_ns

    with torch.inference_mode():
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        logits = adapter.forward(ids, cache)
        state.target_calls += 1
        first = select_token(logits[0, -1], config, forbidden, source)
        released = clock.mark()
        state.committed.append(first)
        state.release_ns.append(released)
        _emit(sink, state, "prefill", 0, cache, None, released - block_start)
        termination = "eos" if first in stop_ids else "budget"

        while len(state.committed) < config.max_new_tokens and termination != "eos":
            block_start = clock.now_ns()
            sequence = state.sequence
            cache.assert_boundary(sequence)
            pending = torch.tensor([[sequence[-1]]], dtype=torch.long, device=device)
            logits = adapter.forward(pending, cache)
            state.target_calls += 1
            token = select_token(logits[0, -1], config, forbidden, source)
            released = clock.mark()
            state.committed.append(token)
            state.release_ns.append(released)
            _emit(sink, state, "target_only", 0, cache, None, released - block_start)
            if token in stop_ids:
                termination = "eos"

    end_ns = clock.mark()
    if options.correctness_checks:
        _assert_termination(state, config, stop_ids, termination)

    return DecodeResult(
        run_id=run_id,
        request_id=request_id,
        output_ids=tuple(state.committed),
        token_release_ns=tuple(state.release_ns),
        termination=termination,  # type: ignore[arg-type]
        start_ns=start_ns,
        first_token_ns=state.release_ns[0],
        last_token_ns=state.release_ns[-1],
        end_ns=end_ns,
        proposed=0,
        accepted=0,
        target_calls=state.target_calls,
        draft_calls=0,
        controller_decisions=0,
        bypass_decisions=0,
        counters={
            "prompt_tokens": len(prompt),
            "blocks": state.blocks,
            "target_cache_length": cache.length,
            # Null rather than zero: this engine never drafts, so a draft cache
            # does not exist to report (SPEC.md section 9.4).
            "draft_cache_length": None,
            "correctness_checks": int(options.correctness_checks),
        },
    )


def _validate_request(prompt: tuple[int, ...], config: DecodeConfig) -> None:
    if not prompt:
        raise ConfigError("prompt must contain at least one token")
    total = len(prompt) + config.max_new_tokens
    if total > MAX_CONTEXT_TOKENS:
        raise ConfigError(
            f"prompt {len(prompt)} + budget {config.max_new_tokens} = {total} exceeds the "
            f"{MAX_CONTEXT_TOKENS}-token context bound"
        )


def _emit(
    sink: EventSink,
    state: RequestState,
    action: str,
    gamma: int,
    cache: CacheHandle,
    draft_cache: CacheHandle | None,
    duration_ns: int,
) -> None:
    sink.emit(
        BlockEvent(
            request_id=state.request_id,
            block_id=state.blocks,
            action=action,  # type: ignore[arg-type]
            gamma=gamma,
            proposed=0,
            accepted=0,
            rejection_position=None,
            committed=1,
            target_calls=state.target_calls,
            draft_calls=state.draft_calls,
            target_cache_after=cache.length,
            draft_cache_after=None if draft_cache is None else draft_cache.length,
            bypass_reason=None,
            duration_ns=duration_ns,
        )
    )
    state.blocks += 1


def _assert_termination(
    state: RequestState,
    config: DecodeConfig,
    stop_ids: set[int],
    termination: str,
) -> None:
    """Invariant I7: the budget and the first committed EOS decide termination."""
    committed = state.committed
    if len(committed) > config.max_new_tokens:
        raise AssertionError(
            f"emitted {len(committed)} tokens past a budget of {config.max_new_tokens}"
        )
    if termination == "eos":
        if committed[-1] not in stop_ids:
            raise AssertionError("terminated on EOS but the last token is not a stop token")
        if any(token in stop_ids for token in committed[:-1]):
            raise AssertionError("emitted tokens after the first committed EOS")
    elif any(token in stop_ids for token in committed):
        raise AssertionError("a stop token was committed without terminating the request")
    if len(state.release_ns) != len(committed):
        raise AssertionError("release timestamps and committed tokens disagree")
