"""The cost controller, driven by fake cost tables so every branch is reachable.

No test here asserts that speculation is fast. They assert that the controller's
arithmetic matches the specification and that it cannot see information it is
not allowed to see. Speed is a milestone 7 measurement, not a correctness gate.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from switchback.controller import (
    CONTEXT_BUCKETS,
    DEFAULT_MARGIN,
    AcceptanceTable,
    ControllerState,
    CostController,
    CostProfile,
    CostTable,
    FixedController,
    context_bucket,
    profile_from_dict,
)
from switchback.types import CONTROLLER_ACTIONS, BlockObservation, ConfigError

# Shapes taken from the measured GB10 profile: a nearly flat target forward and
# a draft about four times cheaper. Values are round numbers, not measurements.
FLAT_TARGET = {1: 45_000_000.0, 2: 45_000_000.0, 3: 45_000_000.0, 5: 45_000_000.0, 9: 46_000_000.0}


def costs(**overrides) -> CostTable:
    base = dict(
        target_forward_ns=dict(FLAT_TARGET),
        draft_forward_ns=11_700_000.0,
        block_overhead_ns=800_000.0,
        draft_prefill_ns=40_000_000.0,
        median_remaining_tokens=120.0,
    )
    base.update(overrides)
    return CostTable(**base)


def acceptance(rate: float, trials: int = 200, positions: int = 8) -> AcceptanceTable:
    return AcceptanceTable(
        successes=tuple(round(rate * trials) for _ in range(positions)),
        trials=tuple(trials for _ in range(positions)),
    )


def profile(rate: float = 0.9, **overrides) -> CostProfile:
    base = dict(acceptance=acceptance(rate), cost=costs())
    base.update(overrides)
    return CostProfile(**base)


def controller(rate: float = 0.9, allow_bypass: bool = True, **overrides) -> CostController:
    instance = CostController(profile=profile(rate, **overrides), allow_bypass=allow_bypass)
    instance.reset()
    return instance


def state(**overrides) -> ControllerState:
    base = dict(cached_tokens=100, remaining_budget=200, draft_initialized=True, blocks_observed=4)
    base.update(overrides)
    return ControllerState(**base)


# --- the four decision regimes the specification names ---------------------


def test_high_acceptance_chooses_a_long_draft() -> None:
    decision = controller(rate=0.9).estimate(state())
    assert decision.action == 8
    assert "beats target-only" in decision.reason


def test_moderate_acceptance_chooses_a_shorter_draft() -> None:
    assert controller(rate=0.8).estimate(state()).action == 4
    assert controller(rate=0.6).estimate(state()).action == 2


def test_low_acceptance_prefers_target_only() -> None:
    decision = controller(rate=0.2).estimate(state())
    assert decision.action == 0
    assert "margin not met" in decision.reason
    assert decision.sticky_bypass is True


def test_the_margin_is_what_rejects_a_marginal_win() -> None:
    """A draft length that wins by less than the margin must not be chosen."""
    strict = controller(rate=0.4)
    decision = strict.estimate(state())
    best = min(
        (item for item in decision.estimates if item.action > 0),
        key=lambda item: item.cost_per_token_ns,
    )
    assert best.cost_per_token_ns < decision.target_only_cost_ns  # it does win
    assert best.cost_per_token_ns > (1 - DEFAULT_MARGIN) * decision.target_only_cost_ns
    assert decision.action == 0
    # With no margin the same profile would draft.
    relaxed = CostController(profile=profile(rate=0.4, margin=0.0))
    relaxed.reset()
    assert relaxed.estimate(state()).action > 0


def test_an_expensive_draft_is_never_worth_it() -> None:
    """If the draft costs as much as the target, drafting cannot win."""
    expensive = CostController(
        profile=CostProfile(acceptance=acceptance(0.99), cost=costs(draft_forward_ns=45_000_000.0))
    )
    expensive.reset()
    assert expensive.estimate(state()).action == 0


def test_a_target_that_scales_with_width_kills_long_drafts() -> None:
    """The flatness of the target forward is measured, not assumed."""
    steep = CostController(
        profile=CostProfile(
            acceptance=acceptance(0.95),
            cost=costs(
                target_forward_ns={
                    1: 45_000_000.0,
                    2: 90_000_000.0,
                    3: 135_000_000.0,
                    5: 225_000_000.0,
                    9: 405_000_000.0,
                }
            ),
        )
    )
    steep.reset()
    assert steep.estimate(state()).action in (0, 1)


# --- budget clipping -------------------------------------------------------


@pytest.mark.parametrize(
    ("remaining", "largest"),
    [(2, 1), (3, 2), (5, 4), (9, 8), (200, 8)],
)
def test_actions_are_clipped_to_the_remaining_budget(remaining: int, largest: int) -> None:
    decision = controller(rate=0.95).estimate(state(remaining_budget=remaining))
    feasible = [item.action for item in decision.estimates if item.feasible and item.action > 0]
    assert max(feasible) == largest
    assert decision.action <= largest


def test_a_budget_of_one_leaves_no_draft_length_feasible() -> None:
    decision = controller(rate=0.95).estimate(state(remaining_budget=1))
    assert decision.action == 0
    assert "no draft length fits" in decision.reason
    # This is a budget fact, not a policy judgement, so it must not stick.
    assert decision.sticky_bypass is False


def test_a_clipped_action_says_why() -> None:
    decision = controller(rate=0.95).estimate(state(remaining_budget=3))
    clipped = [item for item in decision.estimates if not item.feasible]
    assert clipped
    assert all("clipped" in item.reason for item in clipped)
    assert all(math.isinf(item.cost_per_token_ns) for item in clipped)


# --- draft initialization cost --------------------------------------------


def test_startup_cost_is_charged_when_the_draft_is_not_resident() -> None:
    resident = controller(rate=0.9).estimate(state(draft_initialized=True))
    cold = controller(rate=0.9).estimate(state(draft_initialized=False))
    for warm_item, cold_item in zip(resident.estimates, cold.estimates, strict=True):
        if warm_item.action == 0 or not warm_item.feasible:
            continue
        assert cold_item.startup_ns > 0
        assert cold_item.cost_per_token_ns > warm_item.cost_per_token_ns


def test_startup_cost_is_amortized_over_the_shorter_of_budget_and_median() -> None:
    instance = controller(rate=0.9)
    long_request = instance.startup_ns(4, state(draft_initialized=False, remaining_budget=1000))
    short_request = instance.startup_ns(4, state(draft_initialized=False, remaining_budget=10))
    # A short request cannot recover the prefill, so it is charged more per token.
    assert short_request > long_request
    assert long_request == pytest.approx(40_000_000.0 / 120.0)
    assert short_request == pytest.approx(40_000_000.0 / 10.0)


def test_a_very_short_request_declines_to_pay_for_the_draft() -> None:
    instance = CostController(profile=profile(rate=0.9, cost=costs(draft_prefill_ns=400_000_000.0)))
    instance.reset()
    assert instance.estimate(state(draft_initialized=False, remaining_budget=4)).action == 0
    # The same profile drafts happily once the prefill is already paid for.
    assert instance.estimate(state(draft_initialized=True, remaining_budget=4)).action > 0


# --- censoring -------------------------------------------------------------


def test_positions_after_a_rejection_are_censored_not_failed() -> None:
    """The regression this milestone exists to prevent."""
    instance = controller(rate=0.5)
    instance.observe(
        BlockObservation(
            block_id=0,
            action=8,
            proposed=8,
            accepted=2,
            rejection_position=2,
            total_ns=1000,
            committed=3,
            cache_length_after=10,
        )
    )
    table = AcceptanceTable(tuple(instance._successes), tuple(instance._trials))
    assert table.successes[:3] == (1, 1, 0)
    assert table.trials[:3] == (1, 1, 1)
    # Positions 3 through 7 were never evaluated: no trial, no failure.
    assert table.successes[3:] == (0, 0, 0, 0, 0)
    assert table.trials[3:] == (0, 0, 0, 0, 0)


def test_repeated_early_rejections_do_not_collapse_long_position_estimates() -> None:
    instance = controller(rate=0.9)
    before = instance.acceptance().rate(6)
    for index in range(50):
        instance.observe(
            BlockObservation(
                block_id=index,
                action=8,
                proposed=8,
                accepted=0,
                rejection_position=0,
                total_ns=1000,
                committed=1,
                cache_length_after=10,
            )
        )
    after = instance.acceptance()
    # Position 0 moves down, as it should: it really was rejected 50 times.
    # It does not collapse to zero, because 200 calibration trials at 90% are
    # real evidence too and 50 local failures should not erase them.
    assert after.rate(0) < before
    assert after.rate(0) == pytest.approx(180 / 250, abs=0.01)
    # Position 6 keeps the calibration estimate: it was never evaluated at all.
    assert after.rate(6) == pytest.approx(before)


def test_local_evidence_dominates_once_calibration_is_thin() -> None:
    """The converse: weak priors must yield to what this request observed."""
    thin = CostController(
        profile=CostProfile(
            acceptance=AcceptanceTable(successes=(4,) * 8, trials=(5,) * 8), cost=costs()
        )
    )
    thin.reset()
    before = thin.acceptance().rate(0)
    for index in range(50):
        thin.observe(
            BlockObservation(
                block_id=index,
                action=8,
                proposed=8,
                accepted=0,
                rejection_position=0,
                total_ns=1000,
                committed=1,
                cache_length_after=10,
            )
        )
    after = thin.acceptance()
    assert before > 0.6
    assert after.rate(0) < 0.1
    # Still censored: position 6 was never evaluated in this request.
    assert after.rate(6) == pytest.approx(before)


def test_a_fully_accepted_block_counts_every_position_as_a_success() -> None:
    instance = controller(rate=0.5)
    instance.observe(
        BlockObservation(
            block_id=0,
            action=4,
            proposed=4,
            accepted=4,
            rejection_position=None,
            total_ns=1000,
            committed=5,
            cache_length_after=10,
        )
    )
    assert tuple(instance._successes) == (1, 1, 1, 1)
    assert tuple(instance._trials) == (1, 1, 1, 1)


def test_an_unobserved_position_uses_the_prior_not_zero() -> None:
    empty = AcceptanceTable((), ())
    assert empty.rate(0) == 0.5
    assert empty.rate(7) == 0.5
    # Beta(1, 1) smoothing keeps a single observation from being absolute.
    one_success = AcceptanceTable((1,), (1,))
    assert 0.5 < one_success.rate(0) < 1.0
    one_failure = AcceptanceTable((0,), (1,))
    assert 0.0 < one_failure.rate(0) < 0.5


def test_impossible_acceptance_counts_are_refused() -> None:
    with pytest.raises(ConfigError, match="impossible"):
        AcceptanceTable((5,), (2,))
    with pytest.raises(ConfigError, match="negative"):
        AcceptanceTable((-1,), (2,))
    with pytest.raises(ConfigError, match="same length"):
        AcceptanceTable((1, 1), (2,))


# --- sticky bypass ---------------------------------------------------------


def test_bypass_is_sticky_for_the_rest_of_the_request() -> None:
    instance = controller(rate=0.2)
    assert instance.choose(state()) == 0
    assert instance.bypassed
    # Even a state that would otherwise draft keeps returning bypass.
    instance.profile = profile(rate=0.99)
    assert instance.choose(state()) == 0
    assert "sticky bypass" in instance.last_decision.reason


def test_the_no_bypass_ablation_never_sticks() -> None:
    instance = controller(rate=0.2, allow_bypass=False)
    assert instance.choose(state()) == 0
    assert instance.bypassed is False
    instance.profile = profile(rate=0.99)
    assert instance.choose(state()) == 8


def test_a_budget_forced_zero_does_not_trigger_sticky_bypass() -> None:
    instance = controller(rate=0.95)
    assert instance.choose(state(remaining_budget=1)) == 0
    assert instance.bypassed is False
    assert instance.choose(state(remaining_budget=200)) == 8


def test_reset_clears_every_request_local_field() -> None:
    instance = controller(rate=0.2)
    instance.choose(state())
    instance.observe(
        BlockObservation(
            block_id=0,
            action=4,
            proposed=4,
            accepted=1,
            rejection_position=1,
            total_ns=5000,
            committed=2,
            cache_length_after=10,
        )
    )
    assert instance.bypassed and instance._trials and instance.decisions
    instance.reset()
    assert not instance.bypassed
    assert instance._trials == []
    assert instance._successes == []
    assert instance._observed_block_ns == {}
    assert instance.decisions == 0
    assert instance.bypass_decisions == 0
    assert instance.last_decision is None


def test_decision_counters_track_choices() -> None:
    instance = controller(rate=0.9)
    for _ in range(3):
        instance.choose(state())
    assert instance.decisions == 3
    assert instance.bypass_decisions == 0
    instance.profile = profile(rate=0.1)
    instance.choose(state())
    assert instance.decisions == 4
    assert instance.bypass_decisions == 1


# --- what the controller is not allowed to see (invariant I9) --------------


def test_the_state_schema_carries_no_held_out_information() -> None:
    names = {field.name for field in dataclasses.fields(ControllerState)}
    assert names == {
        "cached_tokens",
        "remaining_budget",
        "draft_initialized",
        "blocks_observed",
    }
    forbidden = ("dataset", "label", "prompt", "answer", "cohort", "split", "engine", "expected")
    for name in names:
        assert not any(word in name for word in forbidden), name


def test_a_dataset_label_cannot_even_be_passed() -> None:
    with pytest.raises(TypeError):
        ControllerState(  # type: ignore[call-arg]
            cached_tokens=1,
            remaining_budget=2,
            draft_initialized=True,
            blocks_observed=0,
            dataset="gsm8k",
        )


def test_identical_allowed_state_gives_identical_decisions() -> None:
    """Metamorphic: two requests from different datasets, same allowed state.

    The block observations are made identical on purpose. If a decision differed,
    the difference could only have come from something outside the state schema.
    """
    left = controller(rate=0.8)
    right = controller(rate=0.8)
    observations = [
        BlockObservation(
            block_id=index,
            action=4,
            proposed=4,
            accepted=3,
            rejection_position=3,
            total_ns=90_000_000,
            committed=4,
            cache_length_after=100 + index,
        )
        for index in range(5)
    ]
    for block in observations:
        left.observe(block)
        right.observe(block)
    for cached in (0, 300, 1500, 3000):
        for remaining in (2, 7, 40, 200):
            for initialized in (True, False):
                shared = ControllerState(cached, remaining, initialized, 5)
                assert left.choose(shared) == right.choose(shared)


def test_the_context_bucket_is_derived_not_supplied() -> None:
    assert context_bucket(0) == 0
    assert context_bucket(255) == 0
    assert context_bucket(256) == 1
    assert context_bucket(1024) == 2
    assert context_bucket(5000) == len(CONTEXT_BUCKETS) - 1
    assert state(cached_tokens=1500).context_bucket == 2


def test_invalid_states_are_refused() -> None:
    with pytest.raises(ConfigError, match="cached_tokens"):
        ControllerState(-1, 5, True, 0)
    with pytest.raises(ConfigError, match="remaining_budget"):
        ControllerState(0, 0, True, 0)
    with pytest.raises(ConfigError, match="blocks_observed"):
        ControllerState(0, 5, True, -1)


# --- observed costs --------------------------------------------------------


def test_observed_block_cost_overrides_the_model_with_an_ewma() -> None:
    instance = controller(rate=0.9)
    modelled = instance.block_cost_ns(4, 1.0)
    for index in range(20):
        instance.observe(
            BlockObservation(
                block_id=index,
                action=4,
                proposed=4,
                accepted=4,
                rejection_position=None,
                total_ns=200_000_000,
                committed=5,
                cache_length_after=10,
            )
        )
    assert instance.block_cost_ns(4, 1.0) > modelled
    assert instance.block_cost_ns(4, 1.0) == pytest.approx(200_000_000, rel=0.02)


def test_a_bypass_block_does_not_pollute_the_acceptance_table() -> None:
    instance = controller(rate=0.9)
    instance.observe(
        BlockObservation(
            block_id=0,
            action=0,
            proposed=0,
            accepted=0,
            rejection_position=None,
            total_ns=45_000_000,
            committed=1,
            cache_length_after=10,
        )
    )
    assert instance._trials == []


# --- profile freezing ------------------------------------------------------


def test_the_profile_hash_is_stable_and_sensitive() -> None:
    left = profile(rate=0.9)
    assert left.sha256() == profile(rate=0.9).sha256()
    assert len(left.sha256()) == 64
    assert left.sha256() != profile(rate=0.8).sha256()
    assert (
        left.sha256() != CostProfile(acceptance=acceptance(0.9), cost=costs(), margin=0.10).sha256()
    )


def test_a_profile_round_trips_through_its_dictionary() -> None:
    original = profile(rate=0.85)
    restored = profile_from_dict(original.as_dict())
    assert restored.sha256() == original.sha256()
    assert restored.cost.target_forward_ns == original.cost.target_forward_ns


def test_an_unknown_profile_schema_is_refused() -> None:
    document = profile().as_dict()
    document["schema_version"] = 99
    with pytest.raises(ConfigError, match="schema_version"):
        profile_from_dict(document)


def test_invalid_profiles_are_refused() -> None:
    with pytest.raises(ConfigError, match="bypass"):
        CostProfile(actions=(1, 2, 4))
    with pytest.raises(ConfigError, match="sorted"):
        CostProfile(actions=(0, 4, 2))
    with pytest.raises(ConfigError, match="margin"):
        CostProfile(margin=1.5)
    with pytest.raises(ConfigError, match="ewma_alpha"):
        CostProfile(ewma_alpha=0.0)


def test_invalid_cost_tables_are_refused() -> None:
    with pytest.raises(ConfigError, match="no target forward"):
        CostTable({}, 1.0, 0.0, 1.0, 1.0)
    with pytest.raises(ConfigError, match="draft_forward_ns"):
        CostTable({1: 1.0}, 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ConfigError, match="invalid target forward"):
        CostTable({1: -1.0}, 1.0, 0.0, 1.0, 1.0)


def test_unmeasured_widths_are_interpolated_not_guessed() -> None:
    table = costs(target_forward_ns={1: 10.0, 9: 26.0})
    assert table.target_at_width(1) == 10.0
    assert table.target_at_width(9) == 26.0
    assert table.target_at_width(5) == pytest.approx(18.0)
    # Outside the measured range, clamp rather than extrapolate.
    assert table.target_at_width(50) == 26.0


# --- the fixed baseline ----------------------------------------------------


def test_the_fixed_controller_is_constant_and_budget_aware() -> None:
    fixed = FixedController(4)
    assert fixed.choose(state(remaining_budget=200)) == 4
    assert fixed.choose(state(remaining_budget=3)) == 2
    assert fixed.choose(state(remaining_budget=1)) == 0
    fixed.observe(
        BlockObservation(
            block_id=0,
            action=4,
            proposed=4,
            accepted=0,
            rejection_position=0,
            total_ns=1,
            committed=1,
            cache_length_after=1,
        )
    )
    assert fixed.choose(state(remaining_budget=200)) == 4


def test_the_action_set_matches_the_specification() -> None:
    assert CONTROLLER_ACTIONS == (0, 1, 2, 4, 8)
    assert profile().actions == CONTROLLER_ACTIONS
