import pytest

from src.players.specialist_policy_player import (
    SpecialistPolicyPlayer,
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
        "fish",
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
            opponent_type="fish",
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

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state_factory(),
    )

    assert len(training_agent.episode) == 1

    state, _ = training_agent.episode[0]

    assert state[-1] == 2


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

    assert state[-1] == 6


def test_specialist_player_returns_legal_action(
    training_agent,
    valid_actions,
    round_state_factory,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type="fish",
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
        opponent_type="fish",
        player_name="tested_player",
    )
    player.uuid = "uuid-tested"

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
    assert player.hand_start_stack == 220
    assert player.total_reward_bb == 2.0
    assert training_agent.episode == []


def test_specialist_player_resets_tracking_on_game_start(
    training_agent,
):
    player = SpecialistPolicyPlayer(
        agent=training_agent,
        opponent_type="fish",
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