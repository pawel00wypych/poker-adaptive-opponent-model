from src.agents.q_learning_agent import QLearningAgent
from src.agents.safe_q_player import SafeQPlayer


def sample_round_state(stack=100):
    return {
        "round_count": 1,
        "community_card": [],
        "seats": [
            {
                "name": "safe_q",
                "uuid": "uuid-safe",
                "stack": stack,
                "state": "participating",
            },
            {
                "name": "opponent",
                "uuid": "uuid-opponent",
                "stack": 100,
                "state": "participating",
            },
        ],
        "pot": {
            "main": {
                "amount": 15,
            }
        },
    }


def sample_valid_actions():
    return [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]


def test_safe_q_player_declares_legal_action():
    agent = QLearningAgent(epsilon=0.0)
    player = SafeQPlayer(agent=agent, player_name="safe_q")
    player.uuid = "uuid-safe"

    action, amount = player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=[],
        round_state=sample_round_state(stack=100),
    )

    assert action in {"fold", "call", "raise"}


def test_safe_q_player_updates_reward_after_round():
    agent = QLearningAgent(alpha=0.5, epsilon=0.0)
    player = SafeQPlayer(agent=agent, player_name="safe_q")
    player.uuid = "uuid-safe"

    player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=[],
        round_state=sample_round_state(stack=100),
    )

    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=sample_round_state(stack=120),
    )

    assert player.hands_played == 1
    assert player.total_reward_bb == 2.0