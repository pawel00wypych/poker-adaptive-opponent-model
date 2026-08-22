import pytest

from src.players.learned.fixed_policy_player import FixedPolicyPlayer


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
        "tight",
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


def test_policy_type_does_not_leak_into_the_encoded_state(
    valid_actions,
    round_state_factory,
):
    """Every policy must encode an identical state for identical situations.

    The opponent type used to be part of the state, but each policy owns a
    separate Q-table and always encoded its own type, so the field only ever
    held one value per table. Keeping the policies aligned means their tables
    describe the same situations and can be compared directly.
    """
    from src.agents.monte_carlo_agent import MonteCarloAgent

    encoded_states = {}

    for policy_type in ("unknown", "tight", "aggressive", "calling"):
        agent = MonteCarloAgent(alpha=0.1, epsilon=0.0, epsilon_min=0.0)
        agent.eval()
        player = create_player(agent, policy_type)

        player.declare_action(
            valid_actions=valid_actions,
            hole_card=["HA", "DA"],
            round_state=round_state_factory(),
        )

        assert len(agent.q_table) == 1
        encoded_states[policy_type] = next(iter(agent.q_table))

    distinct_states = set(encoded_states.values())

    assert len(distinct_states) == 1, encoded_states
    assert len(distinct_states.pop()) == 7


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