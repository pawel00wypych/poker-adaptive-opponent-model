import random

from src.players.tight_player import TightPlayer


VALID_ACTIONS = [
    {"action": "fold", "amount": 0},
    {"action": "call", "amount": 10},
    {"action": "raise", "amount": {"min": 20, "max": 100}},
]


def round_state(*, player_stack: int = 100):
    return {
        "street": "preflop",
        "community_card": [],
        "seats": [
            {
                "name": "tight",
                "uuid": "uuid-tight",
                "stack": player_stack,
                "state": "participating",
            },
            {
                "name": "opponent",
                "uuid": "uuid-opponent",
                "stack": 100,
                "state": "participating",
            },
        ],
    }


def test_tight_player_folds_weak_expensive_call():
    player = TightPlayer(rng=random.Random(1))
    player.uuid = "uuid-tight"

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 80},
        ],
        hole_card=["S2", "D7"],
        round_state=round_state(player_stack=100),
    )

    assert action == "fold"
    assert amount == 0


def test_tight_player_can_continue_with_premium_hand():
    player = TightPlayer(rng=random.Random(5))
    player.uuid = "uuid-tight"

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 10},
        ],
        hole_card=["SA", "DA"],
        round_state=round_state(player_stack=100),
    )

    assert action == "call"
    assert amount == 10


def test_tight_player_can_raise_premium_hand_rarely():
    player = TightPlayer(rng=random.Random(123))
    player.uuid = "uuid-tight"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=["SA", "DA"],
        round_state=round_state(player_stack=100),
    )

    assert action == "raise"
    assert amount == 20
