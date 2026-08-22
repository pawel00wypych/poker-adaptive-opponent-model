import pytest

from src.evaluation.constants import STATE_V2_FIELDS
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
    )

    assert state == (
        0,
        PreflopHandEncoder.PREMIUM,
        PokerContextEncoder.NO_PAIR_CONTEXT,
        0,
        3,
        3,
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
    )

    assert state == (
        1,
        HandStrengthEncoder.HIGH_CARD,
        PokerContextEncoder.NO_PAIR_CONTEXT,
        2,
        2,
        0,
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
    )

    assert state == (
        0,
        PreflopHandEncoder.WEAK,
        PokerContextEncoder.NO_PAIR_CONTEXT,
        0,
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
    )

    assert state == (
        2,
        HandStrengthEncoder.HIGH_CARD,
        PokerContextEncoder.NO_PAIR_CONTEXT,
        1,
        0,
        1,
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
    )

    assert state == (
        3,
        HandStrengthEncoder.ONE_PAIR,
        PokerContextEncoder.UNDER_PAIR,
        3,
        2,
        0,
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
    )

    assert state == (
        0,
        PreflopHandEncoder.PREMIUM,
        PokerContextEncoder.NO_PAIR_CONTEXT,
        0,
        0,
        3,
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
    )

    assert state == (
        1,
        HandStrengthEncoder.ONE_PAIR,
        PokerContextEncoder.TOP_PAIR,
        1,
        2,
        2,
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
    )

    assert state == (
        1,
        HandStrengthEncoder.ONE_PAIR,
        PokerContextEncoder.OVERPAIR,
        1,
        2,
        2,
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
    )

    assert state == (
        1,
        HandStrengthEncoder.TWO_PAIR,
        PokerContextEncoder.TWO_PAIR_OR_BETTER,
        1,
        2,
        2,
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
        )


@pytest.mark.parametrize(
    "policy_type",
    ["unknown", "tight", "aggressive", "calling"],
)
def test_state_encoding_is_independent_of_the_active_policy(policy_type):
    """The opponent type was a function of the acting policy, never a signal.

    Each policy owns a separate Q-table and always encoded its own type, so the
    field was constant within any table. Encoding must not depend on which
    policy happens to be acting.
    """
    from src.agents.monte_carlo_agent import MonteCarloAgent
    from src.players.learned.fixed_policy_player import FixedPolicyPlayer

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
    ]
    round_state = {
        "street": "flop",
        "community_card": ["CA", "SK", "D2"],
        "pot": {"main": {"amount": 40}},
        "action_histories": {"flop": []},
        "seats": [
            {
                "name": "tested",
                "uuid": "uuid-tested",
                "stack": 200,
                "state": "participating",
            }
        ],
    }

    agent = MonteCarloAgent(alpha=0.1, epsilon=0.0, epsilon_min=0.0)
    agent.eval()
    player = FixedPolicyPlayer(agent=agent, policy_type=policy_type)
    player.uuid = "uuid-tested"

    captured = {}
    original_encode = StateEncoder.encode

    def capture(**kwargs):
        state = original_encode(**kwargs)
        captured["state"] = state
        return state

    import src.players.learned.fixed_policy_player as module

    module.StateEncoder = type("Spy", (), {"encode": staticmethod(capture)})
    try:
        player.declare_action(valid_actions, ["HA", "DQ"], round_state)
    finally:
        module.StateEncoder = StateEncoder

    assert captured["state"] == (1, 1, 4, 1, 2, 2)


def test_encoded_state_has_six_fields():
    state = StateEncoder.encode(
        player_stack=100,
        valid_actions=[{"action": "call", "amount": 10}],
        round_state={
            "community_card": [],
            "pot": {"main": {"amount": 15}},
        },
        hole_cards=["HA", "DA"],
    )

    assert len(state) == len(STATE_V2_FIELDS) == 6


def test_encode_no_longer_accepts_an_opponent_type():
    with pytest.raises(TypeError):
        StateEncoder.encode(
            player_stack=100,
            valid_actions=[{"action": "call", "amount": 10}],
            round_state={
                "community_card": [],
                "pot": {"main": {"amount": 15}},
            },
            hole_cards=["HA", "DA"],
            opponent_type="calling",
        )