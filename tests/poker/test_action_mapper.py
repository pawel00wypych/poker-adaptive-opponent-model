from src.poker.action_mapper import ActionMapper


def test_call_action_mapping():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    action, amount = ActionMapper.to_engine_action(ActionMapper.CALL, valid_actions)

    assert action == "call"
    assert amount == 10


def test_raise_min_action_mapping():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    action, amount = ActionMapper.to_engine_action(ActionMapper.RAISE_MIN, valid_actions)

    assert action == "raise"
    assert amount == 20


def test_invalid_raise_falls_back_to_call():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": -1, "max": -1}},
    ]

    action, amount = ActionMapper.to_engine_action(ActionMapper.RAISE_MIN, valid_actions)

    assert action == "call"
    assert amount == 10