import pytest

from src.players.specialist_policy_player import (
    SpecialistPolicyPlayer,
)


def start_specialist_round(
    player,
    player_stack=200,
    opponent_stack=200,
):
    player.receive_round_start_message(
        round_count=1,
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


def test_specialist_player_accepts_supported_opponent_type(
    training_agent,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type="calling",
    )

    assert player.opponent_type == "calling"


@pytest.mark.parametrize(
    "opponent_type",
    [
        "tight",
        "aggressive",
        "calling",
    ],
)
def test_specialist_player_accepts_all_supported_types(
    training_agent,
    opponent_type,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type=opponent_type,
    )

    assert player.opponent_type == opponent_type


def test_specialist_player_rejects_unknown_type(
    training_agent,
):
    with pytest.raises(
        ValueError,
        match="Unsupported specialist opponent type",
    ):
        SpecialistPolicyPlayer(
            agent=training_agent,
            opponent_type="unknown",
        )


def test_specialist_player_rejects_invalid_log_interval(
    training_agent,
):
    with pytest.raises(
        ValueError,
        match="log_interval must be greater than zero",
    ):
        SpecialistPolicyPlayer(
            agent=training_agent,
            opponent_type="tight",
            log_interval=0,
        )


def test_specialist_player_encodes_fixed_opponent_type(
    training_agent,
    valid_actions,
    round_state_factory,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type="aggressive",
        player_name="tested_player",
    )
    player.uuid = "uuid-tested"

    start_specialist_round(
        player,
        player_stack=200,
        opponent_stack=200,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state_factory(),
    )

    assert len(training_agent.episode) == 1

    state, _ = training_agent.episode[0]

    assert state[-1] == 1


def test_specialist_player_uses_calling_type_id(
    training_agent,
    valid_actions,
    round_state_factory,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type="calling",
        player_name="tested_player",
    )
    player.uuid = "uuid-tested"

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HK", "DQ"],
        round_state=round_state_factory(),
    )

    state, _ = training_agent.episode[0]

    assert state[-1] == 4


def test_specialist_player_returns_legal_action(
    training_agent,
    valid_actions,
    round_state_factory,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type="tight",
        player_name="tested_player",
    )
    player.uuid = "uuid-tested"

    action, amount = player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state_factory(),
    )

    assert action in {
        "fold",
        "call",
        "raise",
    }

    if action == "fold":
        assert amount == 0
    elif action == "call":
        assert amount == 10
    else:
        assert amount == 20


def test_specialist_player_updates_reward_after_round(
    training_agent,
    valid_actions,
    round_state_factory,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type="tight",
        player_name="tested_player",
    )
    player.uuid = "uuid-tested"

    start_specialist_round(
        player,
        player_stack=200,
        opponent_stack=200,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state_factory(
            player_stack=200,
        ),
    )

    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=round_state_factory(
            player_stack=220,
            opponent_stack=180,
        ),
    )

    assert player.hands_played == 1
    assert player.stack == 220
    assert player.hand_start_stack is None
    assert player.total_reward_bb == 2.0
    assert training_agent.episode == []


def test_specialist_player_resets_tracking_on_game_start(
    training_agent,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type="tight",
    )

    player.initial_stack = 200
    player.hand_start_stack = 180
    player.hands_played = 5
    player.total_reward_bb = -2.0

    player.receive_game_start_message(
        game_info={},
    )

    assert player.initial_stack is None
    assert player.hand_start_stack is None
    assert player.hands_played == 0
    assert player.total_reward_bb == 0.0

def test_specialist_player_reward_includes_blind_paid_before_first_decision(
    training_agent,
    valid_actions,
    round_state_factory,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type="calling",
        player_name="tested_player",
    )
    player.uuid = "uuid-tested"

    start_specialist_round(
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