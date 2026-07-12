from src.players.adaptive_player import AdaptivePlayer
from src.players.rule_based_player import RuleBasedPlayer
from src.experiments.run_agent_comparison import (
    get_classifier_metrics,
    get_hands_played,
)


def test_get_classifier_metrics_returns_zeroes_for_non_adaptive():
    player = RuleBasedPlayer(
        player_name="rule_based"
    )

    metrics = get_classifier_metrics(
        player
    )

    assert metrics == {
        "classified_decisions": 0,
        "correct_classifications": 0,
        "incorrect_classifications": 0,
        "unknown_classifications": 0,
        "classifier_accuracy": 0.0,
        "classifier_coverage": 0.0,
        "policy_switches": 0,
        "first_classification_hand": None,
        "first_correct_classification_hand": None,
        "final_predicted_type": "",
    }


def test_get_classifier_metrics_reads_adaptive_values(
    adaptive_agents,
):
    player = AdaptivePlayer(
        agents=adaptive_agents,
        expected_opponent_type="calling",
    )

    player.classified_decisions = 10
    player.correct_classifications = 8
    player.incorrect_classifications = 2
    player.unknown_classifications = 4
    player.policy_switches = 3
    player.first_classification_hand = 2
    player.first_correct_classification_hand = 4
    player.current_opponent_type = "calling"

    metrics = get_classifier_metrics(
        player
    )

    assert metrics["classified_decisions"] == 10
    assert metrics["correct_classifications"] == 8
    assert metrics["incorrect_classifications"] == 2
    assert metrics["unknown_classifications"] == 4
    assert metrics["classifier_accuracy"] == 0.8
    assert metrics["classifier_coverage"] == 10 / 14
    assert metrics["policy_switches"] == 3
    assert metrics["first_classification_hand"] == 2
    assert metrics["first_correct_classification_hand"] == 4
    assert metrics["final_predicted_type"] == "calling"


def test_get_hands_played_returns_player_value():
    player = RuleBasedPlayer(
        player_name="rule_based"
    )
    player.hands_played = 25

    assert get_hands_played(
        player
    ) == 25


def test_get_hands_played_returns_at_least_one():
    player = RuleBasedPlayer(
        player_name="rule_based"
    )
    player.hands_played = 0

    assert get_hands_played(
        player
    ) == 1