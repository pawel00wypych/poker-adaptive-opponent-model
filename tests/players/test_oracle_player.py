import pytest

from src.players.learned.oracle_player import OraclePlayer


def create_player(
    adaptive_agents,
    oracle_opponent_type: str = "calling",
) -> OraclePlayer:
    player = OraclePlayer(
        agents=adaptive_agents,
        oracle_opponent_type=oracle_opponent_type,
        player_name="tested_player",
    )
    player.uuid = "uuid-tested"

    return player

def start_oracle_round(
    player,
    player_stack: int = 100,
    opponent_stack: int = 100,
    round_count: int = 1,
):
    player.receive_round_start_message(
        round_count=round_count,
        hole_card=["HA", "DA"],
        seats=[
            {
                "name": "tested_player",
                "uuid": "uuid-tested",
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
    )

def test_oracle_requires_all_agents(
    adaptive_agents,
):
    del adaptive_agents["calling"]

    with pytest.raises(
        ValueError,
        match="Missing oracle agents",
    ):
        OraclePlayer(
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
        OraclePlayer(
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
        OraclePlayer(
            agents=adaptive_agents,
            oracle_opponent_type="calling",
            log_interval=0,
        )


@pytest.mark.parametrize(
    "oracle_type",
    ["tight", "aggressive", "calling"],
)
def test_oracle_uses_known_policy_from_first_decision(
    adaptive_agents,
    valid_actions,
    round_state_factory,
    oracle_type,
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

    # The chosen specialist is the only policy that recorded a state; the
    # policy identity lives in which table was used, not inside the state.
    assert len(selected_agent.q_table) == 1

    for policy_type, agent in adaptive_agents.items():
        if policy_type != oracle_type:
            assert len(agent.q_table) == 0


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
    player.active_policy_type = "tight"

    player.receive_game_start_message(
        {}
    )

    assert player.hands_played == 0
    assert player.total_reward_bb == 0.0
    assert player.initial_stack is None
    assert player.hand_start_stack is None
    assert player.policy_usage_counts == {}
    assert player.active_policy_type == "calling"

def test_oracle_player_reward_includes_blind_paid_before_first_decision(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    adaptive_agents["calling"].train()

    player = create_player(
        adaptive_agents,
        oracle_opponent_type="calling",
    )

    start_oracle_round(
        player,
        player_stack=100,
        opponent_stack=100,
    )

    # First decision happens after posting the big blind.
    # The player already has 90 chips, but the hand started at 100.
    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state_factory(
            player_stack=90,
            opponent_stack=100,
        ),
    )

    # Final stack is 80, so reward should be:
    # 80 - 100 = -20 chips = -2 BB.
    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=round_state_factory(
            player_stack=80,
            opponent_stack=120,
        ),
    )

    assert player.hands_played == 1
    assert player.total_reward_bb == -2.0
    assert player.stack == 80
    assert player.hand_start_stack is None
    assert adaptive_agents["calling"].episode == []
