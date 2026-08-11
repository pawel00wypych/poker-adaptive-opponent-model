import pytest

from src.features.hand_strength_encoder import (
    HandStrengthEncoder,
)
from src.features.poker_context_encoder import (
    PokerContextEncoder,
)
from src.features.preflop_hand_encoder import (
    PreflopHandEncoder,
)
from src.features.state_encoder import StateEncoder


def test_preflop_state_encoding_with_premium_hand():
    valid_actions = [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": 10,
        },
        {
            "action": "raise",
            "amount": {
                "min": 20,
                "max": 100,
            },
        },
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
        PokerContextEncoder.NO_PAIR_CONTEXT,
        0,
        3,
        3,
        0,
    )


def test_flop_state_encoding_with_aggressive_opponent():
    valid_actions = [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": 30,
        },
        {
            "action": "raise",
            "amount": {
                "min": 60,
                "max": 100,
            },
        },
    ]

    round_state = {
        "community_card": [
            "HA",
            "D7",
            "C2",
        ],
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
        PokerContextEncoder.NO_PAIR_CONTEXT,
        2,
        2,
        0,
        2,
    )


def test_state_encoding_with_weak_hand():
    valid_actions = [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": 5,
        },
        {
            "action": "raise",
            "amount": {
                "min": 10,
                "max": 100,
            },
        },
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
        opponent_type="tight",
    )

    assert state == (
        0,
        PreflopHandEncoder.WEAK,
        PokerContextEncoder.NO_PAIR_CONTEXT,
        0,
        3,
        3,
        3,
    )


def test_state_encoding_with_free_check():
    valid_actions = [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": 0,
        },
        {
            "action": "raise",
            "amount": {
                "min": 10,
                "max": 100,
            },
        },
    ]

    round_state = {
        "community_card": [
            "HA",
            "D7",
            "C2",
            "S9",
        ],
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
        opponent_type="other",
    )

    assert state == (
        2,
        HandStrengthEncoder.HIGH_CARD,
        PokerContextEncoder.NO_PAIR_CONTEXT,
        1,
        0,
        1,
        4,
    )


def test_state_encoding_on_river():
    valid_actions = [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": 50,
        },
        {
            "action": "raise",
            "amount": {
                "min": 100,
                "max": 200,
            },
        },
    ]

    round_state = {
        "community_card": [
            "HA",
            "D7",
            "C2",
            "S9",
            "HT",
        ],
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
        PokerContextEncoder.UNDER_PAIR,
        3,
        2,
        0,
        3,
    )


def test_state_encoding_without_call_action():
    valid_actions = [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "raise",
            "amount": {
                "min": 20,
                "max": 100,
            },
        },
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
        PokerContextEncoder.NO_PAIR_CONTEXT,
        0,
        0,
        3,
        5,
    )


def test_state_encoding_with_top_pair():
    valid_actions = [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": 10,
        },
        {
            "action": "raise",
            "amount": {
                "min": 20,
                "max": 200,
            },
        },
    ]

    round_state = {
        "community_card": [
            "CA",
            "SK",
            "D2",
        ],
        "pot": {
            "main": {
                "amount": 40,
            }
        },
    }

    state = StateEncoder.encode(
        player_stack=200,
        valid_actions=valid_actions,
        round_state=round_state,
        hole_cards=["HA", "DQ"],
        opponent_type="calling",
    )

    assert state == (
        1,
        HandStrengthEncoder.ONE_PAIR,
        PokerContextEncoder.TOP_PAIR,
        1,
        2,
        2,
        6,
    )


def test_state_encoding_with_overpair():
    valid_actions = [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": 10,
        },
        {
            "action": "raise",
            "amount": {
                "min": 20,
                "max": 200,
            },
        },
    ]

    round_state = {
        "community_card": [
            "CK",
            "S7",
            "D2",
        ],
        "pot": {
            "main": {
                "amount": 40,
            }
        },
    }

    state = StateEncoder.encode(
        player_stack=200,
        valid_actions=valid_actions,
        round_state=round_state,
        hole_cards=["HA", "DA"],
        opponent_type="calling",
    )

    assert state == (
        1,
        HandStrengthEncoder.ONE_PAIR,
        PokerContextEncoder.OVERPAIR,
        1,
        2,
        2,
        6,
    )


def test_state_encoding_with_two_pair_or_better():
    valid_actions = [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": 10,
        },
        {
            "action": "raise",
            "amount": {
                "min": 20,
                "max": 200,
            },
        },
    ]

    round_state = {
        "community_card": [
            "CA",
            "SK",
            "D2",
        ],
        "pot": {
            "main": {
                "amount": 40,
            }
        },
    }

    state = StateEncoder.encode(
        player_stack=200,
        valid_actions=valid_actions,
        round_state=round_state,
        hole_cards=["HA", "DK"],
        opponent_type="calling",
    )

    assert state == (
        1,
        HandStrengthEncoder.TWO_PAIR,
        PokerContextEncoder.TWO_PAIR_OR_BETTER,
        1,
        2,
        2,
        6,
    )


def test_state_encoder_rejects_unknown_opponent_type():
    with pytest.raises(
        ValueError,
        match="Unsupported opponent type",
    ):
        StateEncoder._opponent_type_id(
            "calling_station"
        )


def test_state_encoder_rejects_invalid_community_card_count():
    with pytest.raises(
        ValueError,
        match="Unsupported number of community cards",
    ):
        StateEncoder.encode(
            player_stack=100,
            valid_actions=[
                {
                    "action": "fold",
                    "amount": 0,
                },
                {
                    "action": "call",
                    "amount": 0,
                },
            ],
            round_state={
                "community_card": [
                    "HA",
                    "D7",
                ],
                "pot": {
                    "main": {
                        "amount": 15,
                    }
                },
            },
            hole_cards=["CA", "CK"],
            opponent_type="unknown",
        )


@pytest.mark.parametrize(
    ("opponent_type", "expected_id"),
    [
        ("unknown", 0),
        ("aggressive", 2),
        ("tight", 3),
        ("other", 4),
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