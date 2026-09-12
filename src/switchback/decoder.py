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

from switchback.cache import CacheHandle, rollback_length
from switchback.events import BlockEvent, Clock, EventSink, MonotonicClock, NullSink
from switchback.models.qwen import QwenAdapter
from switchback.sampling import (
    CORRECTION_STREAM,
    PROPOSAL_STREAM,
    BlockOutcome,
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
        termination = "eos" if first in stop_ids else "budget"
        _emit(
            sink,
            state,
            "prefill",
            0,
            cache,
            None,
            released - block_start,
            terminal=termination == "eos",
        )

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
            if token in stop_ids:
                termination = "eos"
            _emit(
                sink,
                state,
                "target_only",
                0,
                cache,
                None,
                released - block_start,
                terminal=termination == "eos",
            )

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
    terminal: bool = False,
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
            terminal=terminal,
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


@dataclass
class DraftState:
    """The draft model's cache and whether it has been initialized yet.

    The draft is prefilled lazily, on the first block that actually drafts, and
    that prefill is charged to the request rather than treated as free
    (SPEC.md section 4.3). A request that ends at its first token never pays it.
    """

    adapter: QwenAdapter
    cache: CacheHandle
    initialized: bool = False


def decode_speculative_greedy(
    target: QwenAdapter,
    draft: QwenAdapter,
    prompt_ids: Sequence[int],
    config: DecodeConfig,
    eos_token_ids: Sequence[int],
    gamma: int,
    request_id: str = "request",
    run_id: str = "run",
    sink: EventSink | None = None,
    clock: Clock | None = None,
    options: EngineOptions | None = None,
) -> DecodeResult:
    """``fixed_g``: greedy speculation at a fixed draft length.

    Follows SPEC.md section 4.3 exactly. Per block:

    1. Feed the pending token to the draft and continue autoregressively until
       ``g`` proposals exist. The draft cache then holds ``S + d[:g-1]``.
    2. Feed ``[S[-1], d_1 ... d_g]`` to the target in **one** forward call of
       width ``g + 1`` against its ``S[:-1]`` cache. Row ``j`` predicts ``d_j+1``
       and row ``g`` predicts the bonus token. There are ``g + 1`` rows, not
       ``g``.
    3. On the first disagreement, commit the matching prefix plus the target's
       own token and crop both caches to ``len(S) + accepted``.
    4. On full acceptance, commit the proposals plus the bonus. The target cache
       is already right; the draft is one behind and needs a catch-up call whose
       cost is counted, not hidden.

    Greedy mode only. The sampled path arrives in milestone 5.
    """
    import torch

    if gamma < 1:
        raise ConfigError(f"speculative decoding requires gamma >= 1, got {gamma}")
    if config.mode != "greedy":
        raise ConfigError("decode_speculative_greedy only implements greedy mode")

    options = options or EngineOptions()
    sink = sink if sink is not None else NullSink()
    clock = clock if clock is not None else MonotonicClock(device=options.device)
    device = options.device

    prompt = tuple(int(value) for value in prompt_ids)
    _validate_request(prompt, config)
    forbidden = forbidden_token_ids(config, eos_token_ids)
    stop_ids = set() if forbidden else {int(value) for value in eos_token_ids}

    state = RequestState(request_id=request_id, run_id=run_id, prompt_ids=prompt)
    target.checks = options.correctness_checks
    draft.checks = options.correctness_checks
    target_cache = target.new_cache()
    drafting = DraftState(adapter=draft, cache=draft.new_cache())
    proposed_total = 0
    accepted_total = 0

    start_ns = clock.mark()
    with torch.inference_mode():
        # --- initialization: prefill, first token, pending-token convention ---
        logits = target.forward(
            torch.tensor([prompt], dtype=torch.long, device=device), target_cache
        )
        state.target_calls += 1
        first = greedy_token(logits[0, -1], forbidden_ids=forbidden)
        released = clock.mark()
        state.committed.append(first)
        state.release_ns.append(released)
        termination = "eos" if first in stop_ids else "budget"
        _emit(
            sink,
            state,
            "prefill",
            0,
            target_cache,
            None,
            released - start_ns,
            terminal=termination == "eos",
        )

        while len(state.committed) < config.max_new_tokens and termination != "eos":
            block_start = clock.now_ns()
            sequence = state.sequence
            target_cache.assert_boundary(sequence)
            remaining = config.max_new_tokens - len(state.committed)

            if remaining == 1:
                # A block emits at least two tokens, so one remaining token is a
                # target-only step. This is a budget rule, not a policy decision.
                pending = torch.tensor([[sequence[-1]]], dtype=torch.long, device=device)
                row = target.forward(pending, target_cache)
                state.target_calls += 1
                token = greedy_token(row[0, -1], forbidden_ids=forbidden)
                released = clock.mark()
                state.committed.append(token)
                state.release_ns.append(released)
                if token in stop_ids:
                    termination = "eos"
                _emit(
                    sink,
                    state,
                    "target_only",
                    0,
                    target_cache,
                    drafting.cache,
                    released - block_start,
                    terminal=termination == "eos",
                )
                continue

            width = min(gamma, remaining - 1)
            if not drafting.initialized:
                # Charged to this request: SPEC.md section 4.3.
                draft.forward(
                    torch.tensor([prompt], dtype=torch.long, device=device), drafting.cache
                )
                state.draft_calls += 1
                drafting.initialized = True

            # The draft must be at the same boundary as the target before it
            # proposes anything. A target-only step does not advance the draft
            # cache, so drafting after one would condition every proposal on a
            # stale prefix. That would not corrupt the output -- the target
            # verifies everything -- but it would quietly destroy acceptance,
            # which is exactly the kind of bug a latency study cannot see.
            drafting.cache.assert_boundary(sequence)

            proposals = _propose_greedy(
                draft, drafting, sequence, width, forbidden, stop_ids, device, state
            )
            proposed_total += len(proposals)

            verification = torch.tensor(
                [[sequence[-1], *proposals]], dtype=torch.long, device=device
            )
            rows = target.forward(verification, target_cache)
            state.target_calls += 1
            outcome = _verify_greedy(rows, proposals, forbidden)
            accepted_total += outcome.accepted

            committed = list(outcome.emitted)
            budget_left = config.max_new_tokens - len(state.committed)
            if len(committed) > budget_left:
                raise AssertionError(
                    f"block produced {len(committed)} tokens with {budget_left} of budget left"
                )
            stopped_at = _first_stop_index(committed, stop_ids)
            if stopped_at is not None:
                committed = committed[: stopped_at + 1]
                termination = "eos"

            released = clock.mark()
            state.committed.extend(committed)
            state.release_ns.extend([released] * len(committed))

            # --- transactional crop: both caches back to S'[:-1] --------------
            keep = rollback_length(len(sequence), outcome.accepted)
            new_length = len(prompt) + len(state.committed) - 1
            if termination == "eos":
                # The tentative suffix beyond the committed EOS is discarded, so
                # this boundary is like any other: both caches end at S'[:-1].
                # Cropping the target to `keep` instead left it one position too
                # long whenever EOS truncated the block. That was found by the
                # trace replay checker, not by any test that existed at the time.
                # The draft may still be one short, because its catch-up call is
                # skipped once termination is known; the block is flagged
                # terminal so a replay can tell that apart from a bug.
                target.crop(target_cache, min(new_length, target.cache_length(target_cache)))
                draft.crop(drafting.cache, min(new_length, draft.cache_length(drafting.cache)))
            else:
                target.crop(target_cache, keep)
                if outcome.all_accepted:
                    # The draft is one candidate behind. Feeding its last
                    # accepted token costs a forward call that must appear in
                    # the counters and in the controller's cost model.
                    draft.forward(
                        torch.tensor([[proposals[-1]]], dtype=torch.long, device=device),
                        drafting.cache,
                    )
                    state.draft_calls += 1
                else:
                    draft.crop(drafting.cache, keep)
                target_cache.assert_boundary(state.sequence)
                drafting.cache.assert_boundary(state.sequence)

            _emit_block(
                sink,
                state,
                gamma=width,
                proposed=len(proposals),
                accepted=outcome.accepted,
                rejection_position=outcome.rejection_position,
                committed=len(committed),
                target_cache=target_cache,
                draft_cache=drafting.cache,
                duration_ns=released - block_start,
                terminal=termination == "eos",
            )

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
        proposed=proposed_total,
        accepted=accepted_total,
        target_calls=state.target_calls,
        draft_calls=state.draft_calls,
        controller_decisions=0,
        bypass_decisions=0,
        counters={
            "prompt_tokens": len(prompt),
            "blocks": state.blocks,
            "gamma": gamma,
            "target_cache_length": target_cache.length,
            "draft_cache_length": drafting.cache.length if drafting.initialized else None,
            "draft_initialized": int(drafting.initialized),
            "correctness_checks": int(options.correctness_checks),
        },
    )


def _propose_greedy(
    draft: QwenAdapter,
    drafting: DraftState,
    sequence: list[int],
    width: int,
    forbidden: Sequence[int],
    stop_ids: set[int],
    device: str,
    state: RequestState,
) -> list[int]:
    """Draw up to ``width`` argmax proposals, feeding each back to the draft.

    A proposed stop token ends drafting early (SPEC.md section 4.3 step 5): there
    is no point proposing past it, since nothing after it could be committed. It
    is still verified by the target, and the actual number of proposals is what
    gets recorded.
    """
    import torch

    proposals: list[int] = []
    pending = sequence[-1]
    for _ in range(width):
        row = draft.forward(
            torch.tensor([[pending]], dtype=torch.long, device=device), drafting.cache
        )
        state.draft_calls += 1
        candidate = greedy_token(row[0, -1], forbidden_ids=forbidden)
        proposals.append(candidate)
        if candidate in stop_ids:
            break
        pending = candidate
    return proposals


def _verify_greedy(rows: Any, proposals: Sequence[int], forbidden: Sequence[int]) -> BlockOutcome:
    """Compare each proposal with the target's argmax at its own causal prefix.

    ``rows`` is ``[1, g + 1, V]``. Row ``j`` was produced from the token at
    position ``j`` of the verification input, so it predicts proposal ``j``; row
    ``g`` predicts the bonus. Using row ``j + 1`` here, or the last row for every
    comparison, is the off-by-one that produces fluent wrong output.
    """
    count = len(proposals)
    if rows.shape[1] != count + 1:
        raise AssertionError(
            f"verification needs {count + 1} rows for {count} proposals, got {rows.shape[1]}"
        )
    for index, candidate in enumerate(proposals):
        expected = greedy_token(rows[0, index], forbidden_ids=forbidden)
        if expected != candidate:
            return BlockOutcome(
                proposed=tuple(proposals),
                accepted=index,
                rejection_position=index,
                emitted=(*proposals[:index], expected),
                all_accepted=False,
            )
    bonus = greedy_token(rows[0, count], forbidden_ids=forbidden)
    return BlockOutcome(
        proposed=tuple(proposals),
        accepted=count,
        rejection_position=None,
        emitted=(*proposals, bonus),
        all_accepted=True,
    )


def _first_stop_index(tokens: Sequence[int], stop_ids: set[int]) -> int | None:
    """Index of the first stop token, or None. Everything after it is dropped."""
    for index, token in enumerate(tokens):
        if token in stop_ids:
            return index
    return None


def _emit_block(
    sink: EventSink,
    state: RequestState,
    gamma: int,
    proposed: int,
    accepted: int,
    rejection_position: int | None,
    committed: int,
    target_cache: CacheHandle,
    draft_cache: CacheHandle,
    duration_ns: int,
    terminal: bool = False,
) -> None:
    sink.emit(
        BlockEvent(
            request_id=state.request_id,
            block_id=state.blocks,
            action="speculative",
            gamma=gamma,
            proposed=proposed,
            accepted=accepted,
            rejection_position=rejection_position,
            committed=committed,
            target_calls=state.target_calls,
            draft_calls=state.draft_calls,
            target_cache_after=target_cache.length,
            draft_cache_after=draft_cache.length,
            bypass_reason=None,
            duration_ns=duration_ns,
            terminal=terminal,
        )
    )
    state.blocks += 1


class _ProposalRecorder:
    """Thin observer that notes which tokens the proposal stream produced.

    Exists so the GPU loop can run :func:`sampling.speculative_block` verbatim --
    the same function the oracle enumerates exhaustively -- instead of a second
    copy of the verification logic that would have to be trusted separately.

    The target's row provider needs the full proposal list before it can issue
    its single batched forward call. ``speculative_block`` draws every proposal
    before it verifies any of them, which is exactly the one-batched-call
    structure of the algorithm, so by the time the first row is requested this
    recorder holds all of them. That ordering is asserted rather than assumed.
    """

    def __init__(self, inner: RandomSource) -> None:
        self.inner = inner
        self.proposals: list[int] = []

    def categorical(self, weights: Any, stream: str) -> int:
        value = self.inner.categorical(weights, stream)
        if stream == PROPOSAL_STREAM:
            self.proposals.append(value)
        return value

    def accept_ratio(self, numerator: float, denominator: float, stream: str) -> bool:
        return self.inner.accept_ratio(numerator, denominator, stream)


def decode_speculative_sampled(
    target: QwenAdapter,
    draft: QwenAdapter,
    prompt_ids: Sequence[int],
    config: DecodeConfig,
    eos_token_ids: Sequence[int],
    gamma: int,
    request_id: str = "request",
    run_id: str = "run",
    source: RandomSource | None = None,
    sink: EventSink | None = None,
    clock: Clock | None = None,
    options: EngineOptions | None = None,
) -> DecodeResult:
    """``fixed_g`` under temperature sampling, using the enumerated sampler.

    Cache handling is identical to the greedy path. The difference is entirely
    in how a candidate is chosen and accepted, and that part is delegated to
    :func:`sampling.speculative_block` so the code exercised here is the code the
    oracle checks in ``tests/unit/test_oracle.py``.

    One deliberate asymmetry with the greedy path: a proposed stop token does
    **not** end drafting early here. The greedy path stops because nothing after
    a stop token could be committed; doing the same in sampled mode would mean
    forking the enumerated function, and the only cost of not doing it is a few
    wasted draft calls in a request that is about to end. It is recorded in the
    counters so the M6 cost model sees the real number of draft calls.
    """
    import torch

    from switchback.sampling import speculative_block

    if gamma < 1:
        raise ConfigError(f"speculative decoding requires gamma >= 1, got {gamma}")
    if config.mode != "sample":
        raise ConfigError("decode_speculative_sampled only implements sample mode")

    options = options or EngineOptions()
    sink = sink if sink is not None else NullSink()
    clock = clock if clock is not None else MonotonicClock(device=options.device)
    device = options.device
    if source is None:
        source = TorchRandomSource(request_seed=config.seed, device=device)

    prompt = tuple(int(value) for value in prompt_ids)
    _validate_request(prompt, config)
    forbidden = forbidden_token_ids(config, eos_token_ids)
    stop_ids = set() if forbidden else {int(value) for value in eos_token_ids}

    state = RequestState(request_id=request_id, run_id=run_id, prompt_ids=prompt)
    target.checks = options.correctness_checks
    draft.checks = options.correctness_checks
    target_cache = target.new_cache()
    drafting = DraftState(adapter=draft, cache=draft.new_cache())
    proposed_total = 0
    accepted_total = 0

    def probabilities(row: Any) -> Any:
        return softmax_probabilities(row, config.temperature, forbidden_ids=forbidden)

    start_ns = clock.mark()
    with torch.inference_mode():
        logits = target.forward(
            torch.tensor([prompt], dtype=torch.long, device=device), target_cache
        )
        state.target_calls += 1
        first = source.categorical(probabilities(logits[0, -1]), CORRECTION_STREAM)
        released = clock.mark()
        state.committed.append(first)
        state.release_ns.append(released)
        termination = "eos" if first in stop_ids else "budget"
        _emit(
            sink,
            state,
            "prefill",
            0,
            target_cache,
            None,
            released - start_ns,
            terminal=termination == "eos",
        )

        while len(state.committed) < config.max_new_tokens and termination != "eos":
            block_start = clock.now_ns()
            sequence = state.sequence
            target_cache.assert_boundary(sequence)
            remaining = config.max_new_tokens - len(state.committed)

            if remaining == 1:
                pending = torch.tensor([[sequence[-1]]], dtype=torch.long, device=device)
                row = target.forward(pending, target_cache)
                state.target_calls += 1
                token = source.categorical(probabilities(row[0, -1]), CORRECTION_STREAM)
                released = clock.mark()
                state.committed.append(token)
                state.release_ns.append(released)
                if token in stop_ids:
                    termination = "eos"
                _emit(
                    sink,
                    state,
                    "target_only",
                    0,
                    target_cache,
                    drafting.cache,
                    released - block_start,
                    terminal=termination == "eos",
                )
                continue

            width = min(gamma, remaining - 1)
            if not drafting.initialized:
                draft.forward(
                    torch.tensor([[*prompt]], dtype=torch.long, device=device), drafting.cache
                )
                state.draft_calls += 1
                drafting.initialized = True

            # See the note in the greedy engine: drafting after a target-only
            # step would condition proposals on a stale prefix.
            drafting.cache.assert_boundary(sequence)

            recorder = _ProposalRecorder(source)
            base = tuple(sequence)
            verification_rows: list[Any] = []

            def draft_row(prefix: tuple[int, ...], _base: tuple[int, ...] = base) -> Any:
                row = draft.forward(
                    torch.tensor([[prefix[-1]]], dtype=torch.long, device=device), drafting.cache
                )
                state.draft_calls += 1
                return probabilities(row[0, -1])

            def target_row(
                prefix: tuple[int, ...],
                _base: tuple[int, ...] = base,
                _width: int = width,
                _rows: list[Any] = verification_rows,
                _recorder: _ProposalRecorder = recorder,
            ) -> Any:
                if not _rows:
                    if len(_recorder.proposals) != _width:
                        raise AssertionError(
                            f"verification began with {len(_recorder.proposals)} of "
                            f"{_width} proposals drawn; the single batched target "
                            f"call would be built from an incomplete block"
                        )
                    batched = torch.tensor(
                        [[_base[-1], *_recorder.proposals]], dtype=torch.long, device=device
                    )
                    logits_rows = target.forward(batched, target_cache)
                    state.target_calls += 1
                    _rows.extend(
                        probabilities(logits_rows[0, index]) for index in range(_width + 1)
                    )
                index = len(prefix) - len(_base)
                if not 0 <= index <= _width:
                    raise AssertionError(f"verification asked for row {index} of {_width + 1}")
                return _rows[index]

            outcome = speculative_block(base, width, target_row, draft_row, recorder)
            proposals = list(outcome.proposed)
            proposed_total += len(proposals)
            accepted_total += outcome.accepted

            committed = list(outcome.emitted)
            budget_left = config.max_new_tokens - len(state.committed)
            if len(committed) > budget_left:
                raise AssertionError(
                    f"block produced {len(committed)} tokens with {budget_left} of budget left"
                )
            stopped_at = _first_stop_index(committed, stop_ids)
            if stopped_at is not None:
                committed = committed[: stopped_at + 1]
                termination = "eos"

            released = clock.mark()
            state.committed.extend(committed)
            state.release_ns.extend([released] * len(committed))

            keep = rollback_length(len(sequence), outcome.accepted)
            new_length = len(prompt) + len(state.committed) - 1
            if termination == "eos":
                # The tentative suffix beyond the committed EOS is discarded, so
                # this boundary is like any other: both caches end at S'[:-1].
                # Cropping the target to `keep` instead left it one position too
                # long whenever EOS truncated the block. That was found by the
                # trace replay checker, not by any test that existed at the time.
                # The draft may still be one short, because its catch-up call is
                # skipped once termination is known; the block is flagged
                # terminal so a replay can tell that apart from a bug.
                target.crop(target_cache, min(new_length, target.cache_length(target_cache)))
                draft.crop(drafting.cache, min(new_length, draft.cache_length(drafting.cache)))
            else:
                target.crop(target_cache, keep)
                if outcome.all_accepted:
                    draft.forward(
                        torch.tensor([[proposals[-1]]], dtype=torch.long, device=device),
                        drafting.cache,
                    )
                    state.draft_calls += 1
                else:
                    draft.crop(drafting.cache, keep)
                target_cache.assert_boundary(state.sequence)
                drafting.cache.assert_boundary(state.sequence)

            _emit_block(
                sink,
                state,
                gamma=width,
                proposed=len(proposals),
                accepted=outcome.accepted,
                rejection_position=outcome.rejection_position,
                committed=len(committed),
                target_cache=target_cache,
                draft_cache=drafting.cache,
                duration_ns=released - block_start,
                terminal=termination == "eos",
            )

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
        proposed=proposed_total,
        accepted=accepted_total,
        target_calls=state.target_calls,
        draft_calls=state.draft_calls,
        controller_decisions=0,
        bypass_decisions=0,
        counters={
            "prompt_tokens": len(prompt),
            "blocks": state.blocks,
            "gamma": gamma,
            "target_cache_length": target_cache.length,
            "draft_cache_length": drafting.cache.length if drafting.initialized else None,
            "draft_initialized": int(drafting.initialized),
            "correctness_checks": int(options.correctness_checks),
        },
    )
