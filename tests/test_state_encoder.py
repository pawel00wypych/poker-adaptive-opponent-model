import pytest

from src.features.hand_strength_encoder import HandStrengthEncoder
from src.features.preflop_hand_encoder import PreflopHandEncoder
from src.features.state_encoder import StateEncoder


def test_preflop_state_encoding_with_premium_hand():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    round_state = {
        "community_card": [],
        "pot": {
            "main": {
                "amount": 15,
            }
        },
    }

    state = StateEncoder.encode(
        player_stack=100,
        valid_actions=valid_actions,
        round_state=round_state,
        hole_cards=["HA", "DA"],
        opponent_type="unknown",
    )

    assert state == (
        0,
        PreflopHandEncoder.PREMIUM,
        2,
        0,
        1,
        0,
    )


def test_flop_state_encoding_with_aggressive_opponent():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 30},
        {"action": "raise", "amount": {"min": 60, "max": 100}},
    ]

    round_state = {
        "community_card": ["HA", "D7", "C2"],
        "pot": {
            "main": {
                "amount": 80,
            }
        },
    }

    state = StateEncoder.encode(
        player_stack=50,
        valid_actions=valid_actions,
        round_state=round_state,
        hole_cards=["H9", "H8"],
        opponent_type="aggressive",
    )

    assert state == (
        1,
        HandStrengthEncoder.HIGH_CARD,
        1,
        2,
        2,
        2,
    )


def test_state_encoding_with_weak_hand():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 5},
        {"action": "raise", "amount": {"min": 10, "max": 100}},
    ]

    round_state = {
        "community_card": [],
        "pot": {
            "main": {
                "amount": 10,
            }
        },
    }

    state = StateEncoder.encode(
        player_stack=100,
        valid_actions=valid_actions,
        round_state=round_state,
        hole_cards=["H7", "D2"],
        opponent_type="fish",
    )

    assert state == (
        0,
        PreflopHandEncoder.WEAK,
        2,
        0,
        1,
        1,
    )


def test_state_encoding_with_free_check():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 0},
        {"action": "raise", "amount": {"min": 10, "max": 100}},
    ]

    round_state = {
        "community_card": ["HA", "D7", "C2", "S9"],
        "pot": {
            "main": {
                "amount": 45,
            }
        },
    }

    state = StateEncoder.encode(
        player_stack=70,
        valid_actions=valid_actions,
        round_state=round_state,
        hole_cards=["SK", "DQ"],
        opponent_type="balanced",
    )

    assert state == (
        2,
        HandStrengthEncoder.HIGH_CARD,
        1,
        1,
        0,
        4,
    )


def test_state_encoding_on_river():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 50},
        {"action": "raise", "amount": {"min": 100, "max": 200}},
    ]

    round_state = {
        "community_card": ["HA", "D7", "C2", "S9", "HT"],
        "pot": {
            "main": {
                "amount": 150,
            }
        },
    }

    state = StateEncoder.encode(
        player_stack=20,
        valid_actions=valid_actions,
        round_state=round_state,
        hole_cards=["C4", "D4"],
        opponent_type="tight",
    )

    assert state == (
        3,
        HandStrengthEncoder.ONE_PAIR,
        0,
        3,
        3,
        3,
    )


def test_state_encoding_without_call_action():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    round_state = {
        "community_card": [],
        "pot": {
            "main": {
                "amount": 15,
            }
        },
    }

    state = StateEncoder.encode(
        player_stack=100,
        valid_actions=valid_actions,
        round_state=round_state,
        hole_cards=["HA", "SK"],
        opponent_type="random",
    )

    assert state == (
        0,
        PreflopHandEncoder.PREMIUM,
        2,
        0,
        0,
        5,
    )

def test_state_encoder_rejects_unknown_opponent_type():
    with pytest.raises(
        ValueError,
        match="Unsupported opponent type",
    ):
        StateEncoder._opponent_type_id(
            "calling_station"
        )


@pytest.mark.parametrize(
    ("opponent_type", "expected_id"),
    [
        ("unknown", 0),
        ("fish", 1),
        ("aggressive", 2),
        ("tight", 3),
        ("balanced", 4),
        ("random", 5),
        ("calling", 6),
    ],
)
def test_state_encoder_maps_supported_opponent_types(
    opponent_type,
    expected_id,
):
    assert (
        StateEncoder._opponent_type_id(
            opponent_type
        )
        == expected_id
    )