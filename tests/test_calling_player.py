from src.players.calling_player import CallingPlayer
from src.players.calling_extreme_player import CallingExtremePlayer


VALID_ACTIONS = [
    {"action": "fold", "amount": 0},
    {"action": "call", "amount": 10},
    {"action": "raise", "amount": {"min": 20, "max": 100}},
]


def test_calling_player_calls_when_call_available():
    player = CallingPlayer()

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state={},
    )

    assert action == "call"
    assert amount == 10


def test_calling_player_checks_when_call_amount_is_zero():
    player = CallingPlayer()

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 0},
            {"action": "raise", "amount": {"min": 10, "max": 100}},
        ],
        hole_card=[],
        round_state={},
    )

    assert action == "call"
    assert amount == 0


def test_calling_player_folds_when_call_not_available():
    player = CallingPlayer()

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "raise", "amount": {"min": 20, "max": 100}},
        ],
        hole_card=[],
        round_state={},
    )

    assert action == "fold"
    assert amount == 0


def test_calling_player_uses_first_action_as_last_fallback():
    player = CallingPlayer()

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "raise", "amount": {"min": 20, "max": 100}},
        ],
        hole_card=[],
        round_state={},
    )

    assert action == "raise"
    assert amount == {"min": 20, "max": 100}


def test_calling_player_tracks_round_results():
    player = CallingPlayer(player_name="calling")
    player.uuid = "uuid-calling"

    round_state_1 = {
        "seats": [
            {"name": "calling", "uuid": "uuid-calling", "stack": 100},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 100},
        ]
    }

    round_state_2 = {
        "seats": [
            {"name": "calling", "uuid": "uuid-calling", "stack": 120},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 80},
        ]
    }

    player.receive_round_result_message([], [], round_state_1)
    player.receive_round_result_message([], [], round_state_2)

    assert player.hands_played == 2
    assert player.total_reward_bb == 2.0


def _round_state_for_calling_extreme(player_stack: int = 100) -> dict:
    return {
        "community_card": [],
        "seats": [
            {"name": "calling_extreme", "uuid": "uuid-calling-extreme", "stack": player_stack},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 100},
        ],
    }


def test_calling_extreme_folds_some_weak_expensive_calls():
    import random

    player = CallingExtremePlayer(rng=random.Random(1))
    player.uuid = "uuid-calling-extreme"

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 80},
        ],
        hole_card=["S2", "D7"],
        round_state=_round_state_for_calling_extreme(player_stack=100),
    )

    assert action == "fold"
    assert amount == 0


def test_calling_extreme_can_continue_with_strong_expensive_hand():
    import random

    player = CallingExtremePlayer(rng=random.Random(5))
    player.uuid = "uuid-calling-extreme"

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 80},
        ],
        hole_card=["HA", "HK"],
        round_state=_round_state_for_calling_extreme(player_stack=100),
    )

    assert action == "call"
    assert amount == 80


def test_calling_extreme_can_value_raise_strong_hand():
    import random

    player = CallingExtremePlayer(rng=random.Random(1), strong_raise_probability=1.0)
    player.uuid = "uuid-calling-extreme"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=["HA", "HK"],
        round_state=_round_state_for_calling_extreme(player_stack=100),
    )

    assert action == "raise"
    assert amount == 20
