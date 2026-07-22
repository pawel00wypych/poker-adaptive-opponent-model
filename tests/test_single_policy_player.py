from src.agents.monte_carlo_agent import MonteCarloAgent
from src.players.single_policy_player import SinglePolicyPlayer


def sample_round_state(stack: int = 100) -> dict:
    return {
        "round_count": 1,
        "community_card": [],
        "seats": [
            {
                "name": "single_policy",
                "uuid": "uuid-single",
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


def sample_valid_actions() -> list[dict]:
    return [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]


def test_single_policy_player_declares_legal_action():
    agent = MonteCarloAgent(epsilon=0.0)

    player = SinglePolicyPlayer(
        agent=agent,
        player_name="single_policy",
    )
    player.uuid = "uuid-single"

    action, amount = player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["HA", "DA"],
        round_state=sample_round_state(stack=100),
    )

    assert action in {"fold", "call", "raise"}

    if action == "fold":
        assert amount == 0
    elif action == "call":
        assert amount == 10
    elif action == "raise":
        assert amount == 20


def test_single_policy_player_updates_positive_reward_after_round():
    agent = MonteCarloAgent(alpha=0.5, epsilon=0.0)

    player = SinglePolicyPlayer(
        agent=agent,
        player_name="single_policy",
    )
    player.uuid = "uuid-single"

    player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["HA", "DA"],
        round_state=sample_round_state(stack=100),
    )

    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=sample_round_state(stack=120),
    )

    assert player.hands_played == 1
    assert player.total_reward_bb == 2.0
    assert player.hand_start_stack == 120


def test_single_policy_player_updates_negative_reward_after_round():
    agent = MonteCarloAgent(alpha=0.5, epsilon=0.0)

    player = SinglePolicyPlayer(
        agent=agent,
        player_name="single_policy",
    )
    player.uuid = "uuid-single"

    player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["H7", "D2"],
        round_state=sample_round_state(stack=100),
    )

    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=sample_round_state(stack=80),
    )

    assert player.hands_played == 1
    assert player.total_reward_bb == -2.0
    assert player.hand_start_stack == 80


def test_single_policy_player_uses_unknown_opponent_state():
    agent = MonteCarloAgent(epsilon=0.0)

    player = SinglePolicyPlayer(
        agent=agent,
        player_name="single_policy",
    )
    player.uuid = "uuid-single"

    player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["HA", "DA"],
        round_state=sample_round_state(stack=100),
    )

    assert len(agent.q_table) == 1

    state = next(iter(agent.q_table))

    assert state[-1] == 0


def test_single_policy_player_resets_tracking_on_game_start():
    agent = MonteCarloAgent(epsilon=0.0)

    player = SinglePolicyPlayer(
        agent=agent,
        player_name="single_policy",
    )

    player.hands_played = 5
    player.total_reward_bb = 4.5
    player.hand_start_stack = 140
    player.initial_stack = 100

    player.receive_game_start_message(game_info={})

    assert player.hands_played == 0
    assert player.total_reward_bb == 0.0
    assert player.hand_start_stack is None
    assert player.initial_stack is None

def test_single_policy_player_accepts_verbose_flag():
    agent = MonteCarloAgent(epsilon=0.0)

    player = SinglePolicyPlayer(
        agent=agent,
        player_name="single_policy_mc",
        verbose=True,
    )

    assert player.verbose is True