import pytest

from src.players.oracle_adaptive_player import OracleAdaptivePlayer


def create_player(
    adaptive_agents,
    oracle_opponent_type: str = "calling",
) -> OracleAdaptivePlayer:
    player = OracleAdaptivePlayer(
        agents=adaptive_agents,
        oracle_opponent_type=oracle_opponent_type,
        player_name="tested_player",
    )
    player.uuid = "uuid-tested"

    return player


def test_oracle_requires_all_agents(
    adaptive_agents,
):
    del adaptive_agents["calling"]

    with pytest.raises(
        ValueError,
        match="Missing oracle agents",
    ):
        OracleAdaptivePlayer(
            agents=adaptive_agents,
            oracle_opponent_type="calling",
        )


def test_oracle_rejects_unknown_opponent_type(
    adaptive_agents,
):
    with pytest.raises(
        ValueError,
        match="Unsupported oracle opponent type",
    ):
        OracleAdaptivePlayer(
            agents=adaptive_agents,
            oracle_opponent_type="unknown",
        )


def test_oracle_rejects_invalid_log_interval(
    adaptive_agents,
):
    with pytest.raises(
        ValueError,
        match="log_interval must be greater than zero",
    ):
        OracleAdaptivePlayer(
            agents=adaptive_agents,
            oracle_opponent_type="calling",
            log_interval=0,
        )


@pytest.mark.parametrize(
    ("oracle_type", "expected_opponent_id"),
    [
        ("fish", 1),
        ("aggressive", 2),
        ("calling", 6),
    ],
)
def test_oracle_uses_known_policy_from_first_decision(
    adaptive_agents,
    valid_actions,
    round_state_factory,
    oracle_type,
    expected_opponent_id,
):
    player = create_player(
        adaptive_agents,
        oracle_opponent_type=oracle_type,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=[
            "HA",
            "DA",
        ],
        round_state=round_state_factory(),
    )

    assert player.active_policy_type == oracle_type
    assert player.final_predicted_type == oracle_type
    assert player.policy_usage_counts[oracle_type] == 1

    selected_agent = adaptive_agents[
        oracle_type
    ]

    assert len(selected_agent.q_table) == 1

    state = next(
        iter(selected_agent.q_table)
    )

    assert state[-1] == expected_opponent_id


def test_oracle_resets_tracking_on_game_start(
    adaptive_agents,
):
    player = create_player(
        adaptive_agents,
        oracle_opponent_type="calling",
    )

    player.hands_played = 5
    player.total_reward_bb = 3.0
    player.initial_stack = 200
    player.hand_start_stack = 150
    player.policy_usage_counts["calling"] = 10
    player.active_policy_type = "fish"

    player.receive_game_start_message(
        {}
    )

    assert player.hands_played == 0
    assert player.total_reward_bb == 0.0
    assert player.initial_stack is None
    assert player.hand_start_stack is None
    assert player.policy_usage_counts == {}
    assert player.active_policy_type == "calling"