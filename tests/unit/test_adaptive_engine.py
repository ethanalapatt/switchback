"""The adaptive engine: the controller drives the same block machinery.

Greedy speculation is exact, so switching draft length mid-request -- or
bypassing entirely -- must not change a single token. These tests pin that, and
pin the bookkeeping the benchmark will report.
"""

from __future__ import annotations

import pytest

from switchback.controller import (
    AcceptanceTable,
    ControllerState,
    CostController,
    CostProfile,
    CostTable,
    FixedController,
)
from switchback.decoder import (
    EngineOptions,
    decode_adaptive_greedy,
    decode_greedy_with_controller,
    decode_target_only,
)
from switchback.events import FakeClock, ListSink
from switchback.types import BlockObservation

from .test_speculation_paths import (
    EOS,
    PROMPT,
    ScriptedAdapter,
    agreeing,
    always_disagreeing,
    cycling,
    disagree_after,
    greedy,
)


def costs(**overrides) -> CostTable:
    base = dict(
        target_forward_ns=dict.fromkeys((1, 2, 3, 5, 9), 45_000_000.0),
        draft_forward_ns=11_700_000.0,
        block_overhead_ns=800_000.0,
        draft_prefill_ns=40_000_000.0,
        median_remaining_tokens=120.0,
    )
    base.update(overrides)
    return CostTable(**base)


def cost_controller(rate: float, allow_bypass: bool = True, trials: int = 200) -> CostController:
    """``trials`` controls how much evidence the calibration prior carries.

    With 200 trials the prior dominates a short request, which is the intended
    behaviour: a handful of local rejections should not erase a well-measured
    calibration. Tests that need the controller to change its mind *inside* one
    request pass a thin prior instead.
    """
    table = AcceptanceTable(
        successes=tuple(round(rate * trials) for _ in range(8)), trials=(trials,) * 8
    )
    instance = CostController(
        profile=CostProfile(acceptance=table, cost=costs()), allow_bypass=allow_bypass
    )
    instance.reset()
    return instance


def run(target, draft, controller, config=None, sink=None, eos=(EOS,)):
    return decode_greedy_with_controller(
        target,
        draft,
        PROMPT,
        config or greedy(max_new_tokens=16),
        list(eos),
        controller,
        sink=sink,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True),
    )


def reference(config=None, eos=(EOS,)):
    return decode_target_only(
        ScriptedAdapter(cycling(0), "target"),
        PROMPT,
        config or greedy(max_new_tokens=16),
        list(eos),
        clock=FakeClock(),
    )


# --- conformance -----------------------------------------------------------


@pytest.mark.parametrize("rate", [0.95, 0.8, 0.5, 0.1])
def test_adaptive_output_matches_target_only_at_every_acceptance_rate(rate: float) -> None:
    """Whatever the controller decides, greedy output is unchanged."""
    target, draft = agreeing()
    result = run(target, draft, cost_controller(rate))
    assert result.output_ids == reference().output_ids


@pytest.mark.parametrize("position", [0, 1, 3])
def test_adaptive_output_matches_when_the_draft_is_wrong(position: int) -> None:
    target, draft = disagree_after(position)
    result = run(target, draft, cost_controller(0.9))
    assert result.output_ids == reference().output_ids


def test_a_controller_that_always_bypasses_reproduces_target_only_exactly() -> None:
    target, draft = agreeing()
    result = run(target, draft, FixedController(0))
    expected = reference()
    assert result.output_ids == expected.output_ids
    assert result.target_calls == expected.target_calls
    assert result.draft_calls == 0
    assert result.proposed == 0


# --- bypass behaviour ------------------------------------------------------


def test_bypass_retires_the_draft_and_never_drafts_again() -> None:
    target, draft = agreeing()
    controller = cost_controller(0.05)
    sink = ListSink()
    result = run(target, draft, controller, sink=sink)
    assert result.draft_calls == 0
    assert draft.calls == []
    assert controller.bypassed
    assert all(event.action != "speculative" for event in sink.events)
    assert result.bypass_decisions == result.controller_decisions > 0


def test_a_bypass_block_records_the_controller_reason_in_the_trace() -> None:
    target, draft = agreeing()
    sink = ListSink()
    run(target, draft, cost_controller(0.05), sink=sink)
    bypassed = [event for event in sink.events if event.bypass_reason]
    assert bypassed
    assert "margin not met" in bypassed[0].bypass_reason
    assert "sticky bypass" in bypassed[-1].bypass_reason


def test_drafting_then_bypassing_mid_request_keeps_the_output_identical() -> None:
    """The controller flips to bypass once the request's own evidence is in."""
    target, draft = always_disagreeing()
    controller = cost_controller(0.95, trials=4)
    sink = ListSink()
    result = run(target, draft, controller, config=greedy(max_new_tokens=24), sink=sink)
    assert result.output_ids == reference(greedy(max_new_tokens=24)).output_ids
    actions = [event.action for event in sink.events]
    # It starts by drafting on the optimistic calibration prior, then the
    # repeated rejections it actually observes push it to bypass.
    assert "speculative" in actions
    assert controller.bypassed
    assert actions.index("target_only") > actions.index("speculative")


def test_the_draft_cache_is_released_on_bypass() -> None:
    target, draft = always_disagreeing()
    controller = cost_controller(0.95, trials=4)
    sink = ListSink()
    run(target, draft, controller, config=greedy(max_new_tokens=24), sink=sink)
    after_bypass = [
        event for event in sink.events if event.action == "target_only" and event.bypass_reason
    ]
    assert after_bypass
    assert all(event.draft_cache_after is None for event in after_bypass)


def test_the_no_bypass_ablation_keeps_drafting() -> None:
    target, draft = always_disagreeing()
    controller = cost_controller(0.95, allow_bypass=False, trials=4)
    sink = ListSink()
    result = run(target, draft, controller, config=greedy(max_new_tokens=24), sink=sink)
    assert result.output_ids == reference(greedy(max_new_tokens=24)).output_ids
    assert controller.bypassed is False
    assert result.draft_calls > 0


# --- counters --------------------------------------------------------------


def test_controller_counters_are_reported() -> None:
    target, draft = agreeing()
    controller = cost_controller(0.95)
    sink = ListSink()
    result = run(target, draft, controller, config=greedy(max_new_tokens=20), sink=sink)
    speculative = [event for event in sink.events if event.action == "speculative"]
    assert result.controller_decisions == len(speculative)
    assert result.bypass_decisions == 0
    assert result.accepted <= result.proposed


def test_the_fixed_engine_reports_no_bypasses() -> None:
    target, draft = agreeing()
    result = run(target, draft, FixedController(4), config=greedy(max_new_tokens=20))
    assert result.bypass_decisions == 0
    assert result.controller_decisions > 0


def test_adaptive_resets_controller_state_between_requests() -> None:
    """Invariant I10: a bypass in request one must not carry into request two."""
    target, draft = always_disagreeing()
    controller = cost_controller(0.95, trials=4)
    config = greedy(max_new_tokens=24)
    first = decode_adaptive_greedy(
        target,
        draft,
        PROMPT,
        config,
        [EOS],
        controller,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True),
    )
    assert controller.bypassed
    second = decode_adaptive_greedy(
        target,
        draft,
        PROMPT,
        config,
        [EOS],
        controller,
        clock=FakeClock(),
        options=EngineOptions(correctness_checks=True),
    )
    assert first.output_ids == second.output_ids
    assert first.draft_calls == second.draft_calls
    assert first.controller_decisions == second.controller_decisions


def test_the_controller_sees_the_real_cache_length_and_budget() -> None:
    """The state the engine builds must describe the engine's actual position."""
    seen: list[ControllerState] = []

    class Recording(FixedController):
        def choose(self, state: ControllerState) -> int:
            seen.append(state)
            return super().choose(state)

        def observe(self, block: BlockObservation) -> None:
            return None

    target, draft = agreeing()
    result = run(target, draft, Recording(2), config=greedy(max_new_tokens=13))
    assert seen
    committed = 1
    for index, state in enumerate(seen):
        assert state.cached_tokens == len(PROMPT) + committed - 1
        assert state.remaining_budget == 13 - committed
        assert state.draft_initialized == (index > 0)
        committed += 3
    assert len(result.output_ids) == 13


@pytest.mark.parametrize("budget", [1, 2, 3, 7, 16])
def test_the_budget_is_met_exactly_under_the_controller(budget: int) -> None:
    target, draft = agreeing()
    result = run(target, draft, cost_controller(0.9), config=greedy(max_new_tokens=budget))
    assert len(result.output_ids) == budget


def test_a_single_token_request_never_asks_the_controller() -> None:
    target, draft = agreeing()
    controller = cost_controller(0.9)
    result = run(target, draft, controller, config=greedy(max_new_tokens=1))
    assert controller.decisions == 0
    assert result.draft_calls == 0
