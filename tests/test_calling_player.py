from src.players.calling_player import CallingPlayer


def test_calling_player_calls_when_call_available():
    player = CallingPlayer()

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    action, amount = player.declare_action(
        valid_actions=valid_actions,
        hole_card=[],
        round_state={},
    )

    assert action == "call"
    assert amount == 10


def test_calling_player_checks_when_call_amount_is_zero():
    player = CallingPlayer()

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 0},
        {"action": "raise", "amount": {"min": 10, "max": 100}},
    ]

    action, amount = player.declare_action(
        valid_actions=valid_actions,
        hole_card=[],
        round_state={},
    )

    assert action == "call"
    assert amount == 0


def test_calling_player_folds_when_call_not_available():
    player = CallingPlayer()

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    action, amount = player.declare_action(
        valid_actions=valid_actions,
        hole_card=[],
        round_state={},
    )

    assert action == "fold"
    assert amount == 0


def test_calling_player_uses_first_action_as_last_fallback():
    player = CallingPlayer()

    valid_actions = [
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    action, amount = player.declare_action(
        valid_actions=valid_actions,
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