from src.features.state_encoder import StateEncoder


def test_preflop_state_encoding():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    round_state = {
        "community_card": [],
        "pot": {
            "main": {
                "amount": 15
            }
        }
    }

    state = StateEncoder.encode(
        player_stack=100,
        valid_actions=valid_actions,
        round_state=round_state,
        opponent_type="unknown",
    )

    assert state == (0, 2, 0, 1, 0)


def test_flop_state_encoding_with_aggressive_opponent():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 30},
        {"action": "raise", "amount": {"min": 60, "max": 100}},
    ]

    round_state = {
        "community_card": ["HA", "D7", "C2"],
        "pot": {
            "main": {
                "amount": 80
            }
        }
    }

    state = StateEncoder.encode(
        player_stack=50,
        valid_actions=valid_actions,
        round_state=round_state,
        opponent_type="aggressive",
    )

    assert state == (1, 1, 2, 2, 2)