import pytest

from src.agents.adaptive_player import AdaptivePlayer
from src.agents.monte_carlo_agent import MonteCarloAgent


def sample_round_state(
    player_stack: int = 100,
    opponent_stack: int = 100,
    community_cards: list[str] | None = None,
    round_count: int = 1,
) -> dict:
    return {
        "round_count": round_count,
        "community_card": community_cards or [],
        "seats": [
            {
                "name": "adaptive_mc",
                "uuid": "uuid-adaptive",
                "stack": player_stack,
                "state": "participating",
            },
            {
                "name": "opponent",
                "uuid": "uuid-opponent",
                "stack": opponent_stack,
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


def test_adaptive_player_verbose_is_disabled_by_default():
    agent = MonteCarloAgent(epsilon=0.0)

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )

    assert player.verbose is False


def test_adaptive_player_accepts_verbose_flag():
    agent = MonteCarloAgent(epsilon=0.0)

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
        verbose=True,
    )

    assert player.verbose is True


def test_adaptive_player_declares_legal_action():
    agent = MonteCarloAgent(epsilon=0.0)

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )
    player.uuid = "uuid-adaptive"

    action, amount = player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["HA", "DA"],
        round_state=sample_round_state(),
    )

    assert action in {"fold", "call", "raise"}

    if action == "fold":
        assert amount == 0
    elif action == "call":
        assert amount == 10
    elif action == "raise":
        assert amount == 20


def test_adaptive_player_updates_opponent_stats():
    agent = MonteCarloAgent(epsilon=0.0)

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )
    player.uuid = "uuid-adaptive"

    player.receive_game_update_message(
        action={
            "player_uuid": "uuid-opponent",
            "action": "raise",
            "amount": 20,
        },
        round_state=sample_round_state(),
    )

    assert player.opponent_stats.raises == 1
    assert player.opponent_stats.total_actions == 1


def test_adaptive_player_ignores_own_action_in_opponent_stats():
    agent = MonteCarloAgent(epsilon=0.0)

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )
    player.uuid = "uuid-adaptive"

    player.receive_game_update_message(
        action={
            "player_uuid": "uuid-adaptive",
            "action": "raise",
            "amount": 20,
        },
        round_state=sample_round_state(),
    )

    assert player.opponent_stats.total_actions == 0


def test_adaptive_player_updates_positive_reward_after_round():
    agent = MonteCarloAgent(
        alpha=0.5,
        epsilon=0.0,
    )

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )
    player.uuid = "uuid-adaptive"

    player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["HA", "DA"],
        round_state=sample_round_state(
            player_stack=100,
            opponent_stack=100,
        ),
    )

    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=sample_round_state(
            player_stack=120,
            opponent_stack=80,
        ),
    )

    assert player.hands_played == 1
    assert player.total_reward_bb == 2.0
    assert player.previous_stack == 120


def test_adaptive_player_updates_negative_reward_after_round():
    agent = MonteCarloAgent(
        alpha=0.5,
        epsilon=0.0,
    )

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )
    player.uuid = "uuid-adaptive"

    player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["H7", "D2"],
        round_state=sample_round_state(
            player_stack=100,
            opponent_stack=100,
        ),
    )

    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=sample_round_state(
            player_stack=80,
            opponent_stack=120,
        ),
    )

    assert player.hands_played == 1
    assert player.total_reward_bb == -2.0
    assert player.previous_stack == 80


def test_adaptive_player_resets_state_on_game_start():
    agent = MonteCarloAgent(epsilon=0.0)

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )

    player.hands_played = 5
    player.total_reward_bb = 4.5
    player.previous_stack = 140
    player.initial_stack = 100
    player.current_opponent_type = "aggressive"

    player.opponent_stats.update_action("raise")
    player.opponent_stats.finish_hand()

    player.receive_game_start_message(game_info={})

    assert player.hands_played == 0
    assert player.total_reward_bb == 0.0
    assert player.previous_stack is None
    assert player.initial_stack is None
    assert player.current_opponent_type == "unknown"
    assert player.opponent_stats.total_actions == 0
    assert player.opponent_stats.hands_observed == 0


def test_adaptive_player_initially_uses_unknown_opponent_type():
    agent = MonteCarloAgent(epsilon=0.0)

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )
    player.uuid = "uuid-adaptive"

    player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["HA", "DA"],
        round_state=sample_round_state(),
    )

    assert player.current_opponent_type == "unknown"
    assert len(agent.q_table) == 1

    state = next(iter(agent.q_table))

    assert state[-1] == 0


def test_adaptive_player_detects_aggressive_opponent_after_enough_actions():
    agent = MonteCarloAgent(epsilon=0.0)

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )
    player.uuid = "uuid-adaptive"

    for _ in range(5):
        player.receive_game_update_message(
            action={
                "player_uuid": "uuid-opponent",
                "action": "raise",
                "amount": 20,
            },
            round_state=sample_round_state(),
        )

    player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["HA", "DA"],
        round_state=sample_round_state(),
    )

    assert player.current_opponent_type == "aggressive"

    state = next(iter(agent.q_table))

    assert state[-1] == 2


def test_adaptive_player_detects_calling_opponent_after_enough_actions():
    agent = MonteCarloAgent(epsilon=0.0)

    player = AdaptivePlayer(
        agent=agent,
        player_name="adaptive_mc",
    )
    player.uuid = "uuid-adaptive"

    actions = [
        "call",
        "call",
        "call",
        "call",
        "fold",
    ]

    for action_name in actions:
        player.receive_game_update_message(
            action={
                "player_uuid": "uuid-opponent",
                "action": action_name,
                "amount": 10,
            },
            round_state=sample_round_state(),
        )

    player.declare_action(
        valid_actions=sample_valid_actions(),
        hole_card=["HK", "DQ"],
        round_state=sample_round_state(),
    )

    assert player.current_opponent_type == "calling"

    state = next(iter(agent.q_table))

    assert state[-1] == 6

def test_adaptive_player_rejects_invalid_log_interval():
    agent = MonteCarloAgent(epsilon=0.0)

    with pytest.raises(
        ValueError,
        match="log_interval must be greater than zero",
    ):
        AdaptivePlayer(
            agent=agent,
            log_interval=0,
        )