import pytest

from src.players.adaptive_player import AdaptivePlayer


def create_player(
    adaptive_agents,
    expected_opponent_type: str = "calling",
    verbose: bool = False,
) -> AdaptivePlayer:
    player = AdaptivePlayer(
        agents=adaptive_agents,
        player_name="tested_player",
        expected_opponent_type=expected_opponent_type,
        verbose=verbose,
    )
    player.uuid = "uuid-tested"

    return player


def send_opponent_actions(
    player: AdaptivePlayer,
    actions: list[str],
    round_state: dict,
) -> None:
    for action_name in actions:
        player.receive_game_update_message(
            action={
                "player_uuid": "uuid-opponent",
                "action": action_name,
                "amount": 10,
            },
            round_state=round_state,
        )


def test_adaptive_player_requires_all_agents(
    adaptive_agents,
):
    del adaptive_agents["calling"]

    with pytest.raises(
        ValueError,
        match="Missing adaptive agents",
    ):
        AdaptivePlayer(
            agents=adaptive_agents,
        )


def test_adaptive_player_rejects_invalid_log_interval(
    adaptive_agents,
):
    with pytest.raises(
        ValueError,
        match="log_interval must be greater than zero",
    ):
        AdaptivePlayer(
            agents=adaptive_agents,
            log_interval=0,
        )


def test_adaptive_player_starts_with_unknown_policy(
    adaptive_agents,
):
    player = create_player(
        adaptive_agents,
    )

    assert player.current_opponent_type == "unknown"
    assert player.active_policy_type == "unknown"


def test_adaptive_player_uses_general_policy_before_classification(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state_factory(),
    )

    assert player.current_opponent_type == "unknown"
    assert player.active_policy_type == "unknown"
    assert player.unknown_classifications == 1
    assert player.classified_decisions == 0
    assert player.policy_usage_counts["unknown"] == 1

    unknown_agent = adaptive_agents["unknown"]

    assert len(unknown_agent.q_table) == 1

    state = next(
        iter(unknown_agent.q_table)
    )

    assert state[-1] == 0


def test_adaptive_player_switches_to_aggressive_policy(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="aggressive",
    )

    round_state = round_state_factory()

    send_opponent_actions(
        player,
        [
            "raise",
            "raise",
            "raise",
            "raise",
            "raise",
        ],
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    assert player.current_opponent_type == "aggressive"
    assert player.active_policy_type == "aggressive"
    assert player.policy_switches == 1
    assert player.correct_classifications == 1
    assert player.incorrect_classifications == 0

    aggressive_agent = adaptive_agents[
        "aggressive"
    ]

    assert len(aggressive_agent.q_table) == 1

    state = next(
        iter(aggressive_agent.q_table)
    )

    assert state[-1] == 2


def test_adaptive_player_switches_to_calling_policy(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="calling",
    )

    round_state = round_state_factory()

    send_opponent_actions(
        player,
        [
            "call",
            "call",
            "call",
            "call",
            "fold",
        ],
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HK", "DQ"],
        round_state=round_state,
    )

    assert player.current_opponent_type == "calling"
    assert player.active_policy_type == "calling"
    assert player.correct_classifications == 1

    state = next(
        iter(
            adaptive_agents[
                "calling"
            ].q_table
        )
    )

    assert state[-1] == 6


def test_adaptive_player_records_incorrect_classification(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="calling",
    )

    round_state = round_state_factory()

    send_opponent_actions(
        player,
        [
            "raise",
            "raise",
            "raise",
            "raise",
            "raise",
        ],
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    assert player.current_opponent_type == "aggressive"
    assert player.correct_classifications == 0
    assert player.incorrect_classifications == 1
    assert player.classifier_accuracy == 0.0


def test_classifier_accuracy_is_one_for_correct_prediction(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="aggressive",
    )

    round_state = round_state_factory()

    send_opponent_actions(
        player,
        ["raise"] * 5,
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    assert player.correct_classifications == 1
    assert player.incorrect_classifications == 0
    assert player.classifier_accuracy == 1.0


def test_classifier_accuracy_ignores_unknown_predictions(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="calling",
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state_factory(),
    )

    assert player.unknown_classifications == 1
    assert player.correct_classifications == 0
    assert player.incorrect_classifications == 0
    assert player.classifier_accuracy == 0.0


def test_classifier_coverage_counts_unknown_and_classified_decisions(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="aggressive",
    )

    round_state = round_state_factory()

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    send_opponent_actions(
        player,
        ["raise"] * 5,
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    assert player.unknown_classifications == 1
    assert player.classified_decisions == 1
    assert player.classifier_coverage == 0.5


def test_policy_switch_is_counted_only_when_policy_changes(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="aggressive",
    )

    round_state = round_state_factory()

    send_opponent_actions(
        player,
        ["raise"] * 5,
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HK", "DK"],
        round_state=round_state,
    )

    assert player.active_policy_type == "aggressive"
    assert player.policy_switches == 1
    assert player.policy_usage_counts["aggressive"] == 2


def test_first_classification_hand_is_recorded(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="aggressive",
    )

    round_state = round_state_factory()

    send_opponent_actions(
        player,
        ["raise"] * 5,
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    assert player.first_classification_hand == 1
    assert player.first_correct_classification_hand == 1


def test_first_correct_classification_is_not_set_for_wrong_prediction(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="calling",
    )

    round_state = round_state_factory()

    send_opponent_actions(
        player,
        ["raise"] * 5,
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    assert player.first_classification_hand == 1
    assert player.first_correct_classification_hand is None


def test_adaptive_player_ignores_own_actions(
    adaptive_agents,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
    )

    player.receive_game_update_message(
        action={
            "player_uuid": "uuid-tested",
            "action": "raise",
            "amount": 20,
        },
        round_state=round_state_factory(),
    )

    assert player.opponent_stats.total_actions == 0


def test_adaptive_player_records_opponent_actions(
    adaptive_agents,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
    )

    player.receive_game_update_message(
        action={
            "player_uuid": "uuid-opponent",
            "action": "call",
            "amount": 10,
        },
        round_state=round_state_factory(),
    )

    assert player.opponent_stats.calls == 1
    assert player.opponent_stats.total_actions == 1


def test_adaptive_player_resets_classifier_statistics(
    adaptive_agents,
):
    player = create_player(
        adaptive_agents,
    )

    player.current_opponent_type = "calling"
    player.active_policy_type = "calling"
    player.classification_counts["calling"] = 5
    player.policy_usage_counts["calling"] = 5
    player.correct_classifications = 5
    player.incorrect_classifications = 2
    player.unknown_classifications = 3
    player.classified_decisions = 7
    player.policy_switches = 2
    player.first_classification_hand = 2
    player.first_correct_classification_hand = 3

    player.receive_game_start_message(
        game_info={},
    )

    assert player.current_opponent_type == "unknown"
    assert player.active_policy_type == "unknown"
    assert player.classification_counts == {}
    assert player.policy_usage_counts == {}
    assert player.correct_classifications == 0
    assert player.incorrect_classifications == 0
    assert player.unknown_classifications == 0
    assert player.classified_decisions == 0
    assert player.policy_switches == 0
    assert player.first_classification_hand is None
    assert player.first_correct_classification_hand is None


def test_final_predicted_type_returns_current_type(
    adaptive_agents,
):
    player = create_player(
        adaptive_agents,
    )

    player.current_opponent_type = "fish"

    assert player.final_predicted_type == "fish"