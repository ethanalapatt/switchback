"""Typed trace events, sinks, and the clock the engine measures with.

Timing rules from SPEC.md section 9.4 that this module exists to enforce:

* CUDA is asynchronous. A host timestamp taken without synchronizing measures
  when work was *queued*, not when it finished, so :class:`MonotonicClock`
  synchronizes before every timestamp that bounds a measured region.
* Token release times reflect what a consumer could actually receive. A verified
  block releases several tokens at once, and they share a timestamp rather than
  being spread evenly across the block.
* The primary timing path uses a minimal in-memory sink. Serialization and any
  UI happen after the timed region.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

BlockAction = Literal["prefill", "target_only", "speculative", "draft_prefill", "draft_catchup"]


class Clock(Protocol):
    """Source of monotonic host timestamps that bound GPU work."""

    def now_ns(self) -> int:
        """Host time after all queued device work for this stream has completed."""

    def mark(self) -> int:
        """Synchronize, then return the host time. Used at measured boundaries."""


@dataclass
class MonotonicClock:
    """Production clock: ``perf_counter_ns`` plus a CUDA synchronization."""

    device: str = "cpu"

    def _synchronize(self) -> None:
        import torch

        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device if self.device != "cuda" else None)

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def mark(self) -> int:
        self._synchronize()
        return time.perf_counter_ns()


@dataclass
class FakeClock:
    """Deterministic clock for boundary tests (invariant I11).

    Every call advances by ``step_ns``, and synchronizations are counted so a
    test can assert that the engine actually synchronized where it claims to.
    """

    step_ns: int = 1000
    current_ns: int = 0
    marks: int = 0

    def now_ns(self) -> int:
        self.current_ns += self.step_ns
        return self.current_ns

    def mark(self) -> int:
        self.marks += 1
        return self.now_ns()


@dataclass(frozen=True)
class BlockEvent:
    """One block of work. Compact by design: no full-vocabulary arrays.

    ``target_cache_after`` and ``draft_cache_after`` are the ledger lengths once
    the block has committed, so a trace can be replayed against the pending-token
    convention without re-running inference.
    """

    request_id: str
    block_id: int
    action: BlockAction
    gamma: int
    proposed: int
    accepted: int
    rejection_position: int | None
    committed: int
    target_calls: int
    draft_calls: int
    target_cache_after: int
    draft_cache_after: int | None
    bypass_reason: str | None
    duration_ns: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventSink(Protocol):
    """Where block events go. Must be cheap enough to sit inside a timed region."""

    def emit(self, event: BlockEvent) -> None: ...


@dataclass
class ListSink:
    """Minimal in-memory sink: one list append per block."""

    events: list[BlockEvent] = field(default_factory=list)

    def emit(self, event: BlockEvent) -> None:
        self.events.append(event)


@dataclass
class NullSink:
    """Discards events. Used when even a list append is unwanted."""

    def emit(self, event: BlockEvent) -> None:
        return None
