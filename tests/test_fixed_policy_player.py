import pytest

from src.players.fixed_policy_player import FixedPolicyPlayer


def create_player(
    eval_agent,
    policy_type: str,
) -> FixedPolicyPlayer:
    player = FixedPolicyPlayer(
        agent=eval_agent,
        policy_type=policy_type,
        player_name="tested_player",
    )
    player.uuid = "uuid-tested"

    return player

def start_fixed_policy_round(
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

@pytest.mark.parametrize(
    "policy_type",
    [
        "unknown",
        "fish",
        "aggressive",
        "calling",
    ],
)
def test_fixed_policy_player_accepts_supported_policy_types(
    eval_agent,
    policy_type,
):
    player = FixedPolicyPlayer(
        agent=eval_agent,
        policy_type=policy_type,
    )

    assert player.policy_type == policy_type


def test_fixed_policy_player_rejects_unsupported_policy_type(
    eval_agent,
):
    with pytest.raises(
        ValueError,
        match="Unsupported policy type",
    ):
        FixedPolicyPlayer(
            agent=eval_agent,
            policy_type="loose_passive",
        )


def test_fixed_policy_player_rejects_invalid_log_interval(
    eval_agent,
):
    with pytest.raises(
        ValueError,
        match="log_interval must be greater than zero",
    ):
        FixedPolicyPlayer(
            agent=eval_agent,
            policy_type="calling",
            log_interval=0,
        )


@pytest.mark.parametrize(
    ("policy_type", "expected_opponent_id"),
    [
        ("unknown", 0),
        ("fish", 1),
        ("aggressive", 2),
        ("calling", 6),
    ],
)
def test_fixed_policy_player_encodes_selected_policy_type(
    eval_agent,
    valid_actions,
    round_state_factory,
    policy_type,
    expected_opponent_id,
):
    player = create_player(
        eval_agent,
        policy_type,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=[
            "HA",
            "DA",
        ],
        round_state=round_state_factory(),
    )

    assert len(eval_agent.q_table) == 1

    state = next(
        iter(eval_agent.q_table)
    )

    assert state[-1] == expected_opponent_id


def test_fixed_policy_player_resets_tracking_on_game_start(
    eval_agent,
):
    player = create_player(
        eval_agent,
        "calling",
    )

    player.hands_played = 10
    player.total_reward_bb = 5.0
    player.initial_stack = 200
    player.hand_start_stack = 150

    player.receive_game_start_message(
        {}
    )

    assert player.hands_played == 0
    assert player.total_reward_bb == 0.0
    assert player.initial_stack is None
    assert player.hand_start_stack is None

def test_fixed_policy_player_reward_includes_blind_paid_before_first_decision(
    training_agent,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        eval_agent=training_agent,
        policy_type="calling",
    )

    start_fixed_policy_round(
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
    assert training_agent.episode == []