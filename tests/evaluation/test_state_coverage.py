"""State-space size and how much of it training actually reaches."""

import pytest

from src.evaluation.constants import STATE_V2_FIELDS
from src.evaluation.diagnostics.state_coverage import (
    STATE_FIELD_BUCKETS,
    describe_state_coverage,
    nominal_state_action_space_size,
    nominal_state_space_size,
)
from src.features.state_encoder import StateEncoder
from src.poker.action_mapper import ActionMapper


def test_bucket_map_matches_the_encoded_state_fields():
    """The documented space must describe the state the encoder emits."""
    assert tuple(STATE_FIELD_BUCKETS) == STATE_V2_FIELDS


def test_nominal_state_space_matches_the_documented_size():
    """Locks the figure quoted in the README and the thesis."""
    assert nominal_state_space_size() == 40_320
    assert nominal_state_action_space_size() == 40_320 * ActionMapper.NUM_ACTIONS


def test_encoder_never_emits_a_value_outside_the_declared_buckets():
    """Guards the bucket map against drifting away from the encoders."""
    situations = []

    boards = [
        [],
        ["HA", "D7", "C2"],
        ["HA", "D7", "C2", "S9"],
        ["HA", "D7", "C2", "S9", "HT"],
    ]
    hands = [["HA", "DA"], ["H7", "D2"], ["CA", "CK"], ["S4", "D4"]]
    call_amounts = [0, 5, 10, 50, 200]
    pots = [0, 15, 40, 80, 300]
    stacks = [5, 20, 100, 400]

    for board in boards:
        for hand in hands:
            for call_amount in call_amounts:
                for pot in pots:
                    for stack in stacks:
                        for small_blind in (False, True):
                            situations.append(
                                StateEncoder.encode(
                                    player_stack=stack,
                                    valid_actions=[
                                        {"action": "fold", "amount": 0},
                                        {
                                            "action": "call",
                                            "amount": call_amount,
                                        },
                                    ],
                                    round_state={
                                        "community_card": board,
                                        "pot": {"main": {"amount": pot}},
                                    },
                                    hole_cards=hand,
                                    is_small_blind=small_blind,
                                )
                            )

    limits = list(STATE_FIELD_BUCKETS.values())

    for state in situations:
        assert len(state) == len(limits)
        for value, limit, field in zip(state, limits, STATE_V2_FIELDS):
            assert 0 <= value < limit, (field, value, limit)


def test_coverage_counts_only_states_with_recorded_visits():
    """Reading a Q-table inserts the state, so keys overstate what was learned."""
    q_table = {
        (0, 0, 0, 0, 0, 0, 0): [1.0, 0.0, 0.0],
        (1, 1, 1, 1, 1, 1, 1): [0.0, 0.0, 0.0],
    }
    visit_counts = {
        (0, 0, 0, 0, 0, 0, 0): [3, 1, 0],
        (1, 1, 1, 1, 1, 1, 1): [0, 0, 0],
    }

    coverage = describe_state_coverage(q_table, visit_counts)

    assert coverage.table_entries == 2
    assert coverage.reached_states == 1
    assert coverage.reached_state_actions == 2
    assert coverage.mean_actions_tried_per_reached_state == pytest.approx(2.0)


def test_coverage_is_zero_for_an_untouched_table():
    coverage = describe_state_coverage({}, {})

    assert coverage.reached_states == 0
    assert coverage.state_coverage == 0.0
    assert coverage.state_action_coverage == 0.0
    assert coverage.mean_actions_tried_per_reached_state == 0.0


def test_coverage_ratios_use_the_nominal_space():
    q_table = {(0, 0, 0, 0, 0, 0, 0): [1.0, 0.0, 0.0]}
    visit_counts = {(0, 0, 0, 0, 0, 0, 0): [1, 0, 0]}

    coverage = describe_state_coverage(q_table, visit_counts)

    assert coverage.state_coverage == pytest.approx(
        1 / nominal_state_space_size()
    )
    assert coverage.state_action_coverage == pytest.approx(
        1 / nominal_state_action_space_size()
    )


def test_coverage_serialises_for_reports():
    coverage = describe_state_coverage(
        {(0, 0, 0, 0, 0, 0, 0): [1.0, 0.0, 0.0]},
        {(0, 0, 0, 0, 0, 0, 0): [2, 0, 0]},
    )

    payload = coverage.to_dict()

    assert payload["reached_states"] == 1
    assert payload["nominal_states"] == nominal_state_space_size()
