from src.agents.rule_based_player import RuleBasedPlayer


def test_rule_based_player_calls_when_call_is_free():
    player = RuleBasedPlayer()

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 0},
        {"action": "raise", "amount": {"min": 10, "max": 100}},
    ]

    action, amount = player.declare_action(valid_actions, hole_card=[], round_state={})

    assert action == "call"
    assert amount == 0


def test_rule_based_player_folds_expensive_call():
    player = RuleBasedPlayer()

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 50},
        {"action": "raise", "amount": {"min": 80, "max": 100}},
    ]

    action, amount = player.declare_action(valid_actions, hole_card=[], round_state={})

    assert action == "fold"
    assert amount == 0


def test_rule_based_player_calls_cheap_call():
    player = RuleBasedPlayer()

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    action, amount = player.declare_action(valid_actions, hole_card=[], round_state={})

    assert action == "call"
    assert amount == 10

def test_rule_based_player_tracks_round_results():
    player = RuleBasedPlayer(player_name="rule_based")
    player.uuid = "uuid-rule"

    round_state_1 = {
        "seats": [
            {"name": "rule_based", "uuid": "uuid-rule", "stack": 100},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 100},
        ]
    }

    round_state_2 = {
        "seats": [
            {"name": "rule_based", "uuid": "uuid-rule", "stack": 70},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 130},
        ]
    }

    player.receive_round_result_message([], [], round_state_1)
    player.receive_round_result_message([], [], round_state_2)

    assert player.hands_played == 2
    assert player.total_reward_bb == -3.0