from src.agents.calling_player import CallingPlayer


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