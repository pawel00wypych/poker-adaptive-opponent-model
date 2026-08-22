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

    action, amount = ActionMapper.to_engine_action(
        ActionMapper.RAISE_MIN, valid_actions
    )

    assert action == "raise"
    assert amount == 20


def test_invalid_raise_falls_back_to_call():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": -1, "max": -1}},
    ]

    action, amount = ActionMapper.to_engine_action(
        ActionMapper.RAISE_MIN, valid_actions
    )

    assert action == "call"
    assert amount == 10

FREE_CHECK_ACTIONS = [
    {"action": "fold", "amount": 0},
    {"action": "call", "amount": 0},
    {"action": "raise", "amount": {"min": 10, "max": 200}},
]

PAID_CALL_ACTIONS = [
    {"action": "fold", "amount": 0},
    {"action": "call", "amount": 10},
    {"action": "raise", "amount": {"min": 20, "max": 200}},
]


def test_fold_is_not_legal_when_calling_is_free():
    legal = ActionMapper.get_legal_action_ids(FREE_CHECK_ACTIONS)

    assert ActionMapper.FOLD not in legal
    assert legal == [ActionMapper.CALL, ActionMapper.RAISE_MIN]


def test_fold_remains_legal_when_calling_costs_chips():
    legal = ActionMapper.get_legal_action_ids(PAID_CALL_ACTIONS)

    assert ActionMapper.FOLD in legal


def test_fold_action_id_is_redirected_to_the_free_check():
    action, amount = ActionMapper.to_engine_action(
        ActionMapper.FOLD, FREE_CHECK_ACTIONS
    )

    assert action == "call"
    assert amount == 0


def test_fold_action_id_still_folds_when_calling_costs_chips():
    action, amount = ActionMapper.to_engine_action(
        ActionMapper.FOLD, PAID_CALL_ACTIONS
    )

    assert action == "fold"


def test_legal_actions_are_never_empty_with_only_a_free_check():
    legal = ActionMapper.get_legal_action_ids(
        [{"action": "fold", "amount": 0}, {"action": "call", "amount": 0}]
    )

    assert legal == [ActionMapper.CALL]
