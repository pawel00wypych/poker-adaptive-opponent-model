from src.players.always_call_player import AlwaysCallPlayer


def test_always_call_player_calls_when_call_available():
    player = AlwaysCallPlayer()

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 10},
            {"action": "raise", "amount": {"min": 20, "max": 100}},
        ],
        hole_card=[],
        round_state={},
    )

    assert action == "call"
    assert amount == 10


def test_always_call_player_checks_when_call_amount_is_zero():
    player = AlwaysCallPlayer()

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 0},
            {"action": "raise", "amount": {"min": 20, "max": 100}},
        ],
        hole_card=[],
        round_state={},
    )

    assert action == "call"
    assert amount == 0


def test_always_call_player_folds_when_call_unavailable():
    player = AlwaysCallPlayer()

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


def test_always_call_player_tracks_round_results():
    player = AlwaysCallPlayer(player_name="always_call")
    player.uuid = "uuid-always-call"

    round_state_1 = {
        "seats": [
            {"name": "always_call", "uuid": "uuid-always-call", "stack": 100},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 100},
        ]
    }

    round_state_2 = {
        "seats": [
            {"name": "always_call", "uuid": "uuid-always-call", "stack": 80},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 120},
        ]
    }

    player.receive_round_result_message([], [], round_state_1)
    player.receive_round_result_message([], [], round_state_2)

    assert player.hands_played == 2
    assert player.total_reward_bb == -2.0
