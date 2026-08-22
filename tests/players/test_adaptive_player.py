import pytest

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.players.learned.adaptive_player import AdaptivePlayer


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

def start_adaptive_round(
    player: AdaptivePlayer,
    player_stack: int = 200,
    opponent_stack: int = 200,
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

    assert len(state) == 7


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

    assert len(state) == 7


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

    assert len(state) == 7


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


def test_classifier_coverage_excludes_unknown_and_other_decisions(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    """Only decisions that select a specialist count as covered."""
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
        ["fold"] + ["call"] * 7 + ["raise"] * 2,
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    send_opponent_actions(
        player,
        ["raise"] * 3,
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    assert player.unknown_classifications == 1
    assert player.other_classifications == 1
    assert player.classified_decisions == 1
    assert player.classifier_coverage == pytest.approx(1 / 3)


def _drive_other_classification(
    player,
    valid_actions,
    round_state,
):
    """Feed an action mix that matches no known opponent family."""
    send_opponent_actions(
        player,
        ["fold"] + ["call"] * 7 + ["raise"] * 2,
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )


def test_other_classification_does_not_count_as_covered(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    """Previously 'other' inflated coverage to 100% with no specialist used."""
    player = create_player(
        adaptive_agents,
        expected_opponent_type="calling",
    )
    round_state = round_state_factory()

    _drive_other_classification(player, valid_actions, round_state)

    assert player.current_opponent_type == "other"
    assert player.other_classifications == 1
    assert player.classified_decisions == 0
    assert player.classifier_coverage == 0.0


def test_other_classification_does_not_affect_accuracy(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="calling",
    )
    round_state = round_state_factory()

    _drive_other_classification(player, valid_actions, round_state)

    assert player.correct_classifications == 0
    assert player.incorrect_classifications == 0
    assert player.classifier_accuracy == 0.0


def test_other_classification_selects_the_general_policy(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    """Behaviour is unchanged; only the metric was wrong."""
    player = create_player(
        adaptive_agents,
        expected_opponent_type="calling",
    )
    round_state = round_state_factory()

    _drive_other_classification(player, valid_actions, round_state)

    assert player.active_policy_type == "unknown"


def test_other_rate_reports_unmatched_decisions(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="calling",
    )
    round_state = round_state_factory()

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )
    _drive_other_classification(player, valid_actions, round_state)

    assert player.other_rate == pytest.approx(0.5)


def test_other_counter_resets_between_games(
    adaptive_agents,
    valid_actions,
    round_state_factory,
):
    player = create_player(
        adaptive_agents,
        expected_opponent_type="calling",
    )
    _drive_other_classification(
        player,
        valid_actions,
        round_state_factory(),
    )
    assert player.other_classifications == 1

    player.receive_game_start_message(None)

    assert player.other_classifications == 0
    assert player.other_rate == 0.0


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

    player.current_opponent_type = "tight"

    assert player.final_predicted_type == "tight"

def create_training_adaptive_agents() -> dict[str, MonteCarloAgent]:
    agents = {}

    for opponent_type in [
        "unknown",
        "tight",
        "aggressive",
        "calling",
    ]:
        agent = MonteCarloAgent(
            alpha=1.0,
            epsilon=0.0,
            epsilon_min=0.0,
            alpha_mode="constant",
        )
        agent.train()
        agents[opponent_type] = agent

    return agents


def test_adaptive_player_tracks_reward_between_consecutive_round_results(
    valid_actions,
    round_state_factory,
):
    agents = create_training_adaptive_agents()
    player = create_player(
        agents,
        expected_opponent_type="aggressive",
    )

    start_adaptive_round(
        player,
        player_stack=200,
        opponent_stack=200,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state_factory(
            player_stack=200,
            opponent_stack=200,
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
    assert player.total_reward_bb == 2.0
    assert player.stack == 220
    assert player.hand_start_stack is None

    start_adaptive_round(
        player,
        player_stack=220,
        opponent_stack=180,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HK", "DK"],
        round_state=round_state_factory(
            player_stack=220,
            opponent_stack=180,
            round_count=2,
        ),
    )

    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=round_state_factory(
            player_stack=190,
            opponent_stack=210,
            round_count=2,
        ),
    )

    assert player.hands_played == 2
    assert player.total_reward_bb == -1.0
    assert player.hand_start_stack is None


def test_adaptive_player_updates_all_policies_that_acted_before_switch(
    valid_actions,
    round_state_factory,
):
    agents = create_training_adaptive_agents()
    player = create_player(
        agents,
        expected_opponent_type="aggressive",
    )

    start_adaptive_round(
        player,
        player_stack=200,
        opponent_stack=200,
    )

    round_state = round_state_factory(
        player_stack=200,
        opponent_stack=200,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HA", "DA"],
        round_state=round_state,
    )

    assert player.active_policy_type == "unknown"
    assert len(agents["unknown"].episode) == 1

    unknown_state, unknown_action_id = agents["unknown"].episode[0]

    send_opponent_actions(
        player,
        ["raise"] * 5,
        round_state,
    )

    player.declare_action(
        valid_actions=valid_actions,
        hole_card=["HK", "DK"],
        round_state=round_state,
    )

    assert player.active_policy_type == "aggressive"
    assert player.policy_switches == 1
    assert len(agents["aggressive"].episode) == 1

    aggressive_state, aggressive_action_id = agents["aggressive"].episode[0]

    # Both policies describe the same situation now that the policy identity
    # is no longer part of the state.
    assert unknown_state == aggressive_state

    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=round_state_factory(
            player_stack=230,
            opponent_stack=170,
        ),
    )

    assert agents["unknown"].episode == []
    assert agents["aggressive"].episode == []

    assert unknown_state in agents["unknown"].q_table
    assert aggressive_state in agents["aggressive"].q_table

    assert agents["unknown"].q_table[unknown_state][unknown_action_id] == 3.0
    assert agents["aggressive"].q_table[aggressive_state][aggressive_action_id] == 3.0

    assert player.hands_played == 1
    assert player.total_reward_bb == 3.0
    assert player.stack == 230
    assert player.hand_start_stack is None
