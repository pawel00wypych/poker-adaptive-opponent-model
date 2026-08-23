import pandas as pd
import pytest

from src.evaluation.metrics.evaluation_metrics import (
    calculate_final_model_metrics,
)
from src.evaluation.metrics.seed_statistics import (
    SEED_CI_LOWER_COLUMN,
    SEED_CI_UPPER_COLUMN,
    SEED_STATISTIC_COLUMNS,
)
from src.evaluation.reporting.report_descriptions import METRIC_DESCRIPTIONS
from src.evaluation.reporting.training_opponent_report import aggregate_across_seeds

# Pinned as literals rather than imported constants: these names appear in the
# result CSVs, so a test that followed the constant would not notice the value
# changing underneath it.
GAME_STANDARD_ERROR = "game_standard_error"
GAME_CI_LOWER = "game_ci_95_lower"
GAME_CI_UPPER = "game_ci_95_upper"
GAME_LEVEL_COLUMNS = (GAME_STANDARD_ERROR, GAME_CI_LOWER, GAME_CI_UPPER)


def test_calculate_final_model_metrics_groups_by_seed_and_training_episode(
    tmp_path,
):
    csv_path = tmp_path / "final_model_eval.csv"

    df = pd.DataFrame(
        [
            {
                "training_run": "state_v2",
                "model_seed": 42,
                "model_source": "final",
                "training_episode": 5000,
                "checkpoint_episode": None,
                "experiment_id": "seed_42_episodes_5000",
                "experiment_name": "adaptive_mc_vs_calling",
                "game_id": 0,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "final_stack": 220,
                "initial_stack": 200,
                "profit": 20,
                "profit_bb": 2,
                "hands_played": 10,
                "won_game": 1,
                "busted": 0,
                "ended_by_bust": 1,
                "ended_by_round_limit": 0,
                "classified_decisions": 10,
                "correct_classifications": 10,
                "incorrect_classifications": 0,
                "unknown_classifications": 2,
                "other_classifications": 0,
                "classifier_accuracy": 1.0,
                "classifier_coverage": 10 / 12,
                "policy_switches": 1,
                "first_classification_hand": 2,
                "first_correct_classification_hand": 2,
                "first_classification_action_count": 5,
                "first_correct_classification_action_count": 5,
                "final_predicted_type": "calling",
                "policy_decisions": 0,
                "unseen_state_decisions": 0,
                "untried_action_selections": 0,
                "unseen_state_decision_rate": 0.0,
                "untried_action_selection_rate": 0.0,
            },
            {
                "training_run": "state_v2",
                "model_seed": 42,
                "model_source": "final",
                "training_episode": 5000,
                "checkpoint_episode": None,
                "experiment_id": "seed_42_episodes_5000",
                "experiment_name": "adaptive_mc_vs_calling",
                "game_id": 1,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "final_stack": 180,
                "initial_stack": 200,
                "profit": -20,
                "profit_bb": -2,
                "hands_played": 10,
                "won_game": 0,
                "busted": 0,
                "ended_by_bust": 0,
                "ended_by_round_limit": 1,
                "classified_decisions": 8,
                "correct_classifications": 6,
                "incorrect_classifications": 2,
                "unknown_classifications": 4,
                "other_classifications": 0,
                "classifier_accuracy": 0.75,
                "classifier_coverage": 8 / 12,
                "policy_switches": 2,
                "first_classification_hand": 3,
                "first_correct_classification_hand": 4,
                "first_classification_action_count": 6,
                "first_correct_classification_action_count": 7,
                "final_predicted_type": "calling",
                "policy_decisions": 0,
                "unseen_state_decisions": 0,
                "untried_action_selections": 0,
                "unseen_state_decision_rate": 0.0,
                "untried_action_selection_rate": 0.0,
            },
        ]
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    result = calculate_final_model_metrics(str(csv_path))

    assert len(result) == 1

    row = result.iloc[0]

    assert row["model_seed"] == 42
    assert row["training_episode"] == 5000
    assert row["games"] == 2
    assert row["total_profit_bb"] == 0
    assert row["total_hands"] == 20
    assert row["bb_per_100"] == 0
    assert row["global_classifier_accuracy"] == pytest.approx(88.88888888888889)

    assert row["global_classifier_coverage"] == pytest.approx(75.0)


def test_global_classifier_coverage_excludes_other_classifications(tmp_path):
    """'other' is an unconverted opportunity, so it lowers coverage."""
    csv_path = tmp_path / "other_classifications.csv"

    row = {
        "training_run": "state_v2",
        "model_seed": 42,
        "model_source": "final",
        "training_episode": 5000,
        "checkpoint_episode": None,
        "experiment_id": "seed_42_episodes_5000",
        "experiment_name": "adaptive_mc_vs_calling",
        "game_id": 0,
        "agent_name": "adaptive_mc",
        "opponent_name": "calling",
        "final_stack": 220,
        "initial_stack": 200,
        "profit": 20,
        "profit_bb": 2,
        "hands_played": 10,
        "won_game": 1,
        "busted": 0,
        "ended_by_bust": 1,
        "ended_by_round_limit": 0,
        "classified_decisions": 5,
        "correct_classifications": 5,
        "incorrect_classifications": 0,
        "unknown_classifications": 3,
        "other_classifications": 2,
        "classifier_accuracy": 1.0,
        "classifier_coverage": 0.5,
        "policy_switches": 1,
        "first_classification_hand": 2,
        "first_correct_classification_hand": 2,
        "first_classification_action_count": 5,
        "first_correct_classification_action_count": 5,
        "final_predicted_type": "calling",
        "policy_decisions": 0,
        "unseen_state_decisions": 0,
        "untried_action_selections": 0,
        "unseen_state_decision_rate": 0.0,
        "untried_action_selection_rate": 0.0,
    }

    pd.DataFrame([row]).to_csv(csv_path, index=False)

    result = calculate_final_model_metrics(str(csv_path))
    metrics = result.iloc[0]

    assert metrics["total_other_classifications"] == 2
    # 5 covered out of 5 + 3 unknown + 2 other, not 5 / 8.
    assert metrics["global_classifier_coverage"] == pytest.approx(50.0)
    assert metrics["global_other_rate"] == pytest.approx(20.0)
    # Accuracy is scored only over committed specialist classifications.
    assert metrics["global_classifier_accuracy"] == pytest.approx(100.0)


def test_final_model_metrics_ignore_untrained_baseline_replicates(tmp_path):
    csv_path = tmp_path / "mixed_head_to_head.csv"
    final_row = {
        "training_run": "state_v2",
        "model_seed": 42,
        "model_source": "final",
        "training_episode": 5000,
        "checkpoint_episode": None,
        "evaluation_replicate_id": None,
        "experiment_name": "adaptive_mc_vs_rule_based",
        "game_id": 0,
        "agent_name": "adaptive_mc",
        "opponent_name": "rule_based",
        "profit_bb": 2.0,
        "hands_played": 10,
        "won_game": 1,
        "busted": 0,
        "ended_by_bust": 0,
        "ended_by_round_limit": 1,
        "classified_decisions": 0,
        "correct_classifications": 0,
        "incorrect_classifications": 0,
        "unknown_classifications": 0,
        "other_classifications": 0,
        "classifier_accuracy": 0.0,
        "classifier_coverage": 0.0,
        "policy_switches": 0,
        "first_classification_hand": None,
        "first_correct_classification_hand": None,
        "first_classification_action_count": None,
        "first_correct_classification_action_count": None,
        "policy_decisions": 0,
        "unseen_state_decisions": 0,
        "untried_action_selections": 0,
        "unseen_state_decision_rate": 0.0,
        "untried_action_selection_rate": 0.0,
    }
    baseline_row = {
        **final_row,
        "training_run": None,
        "model_seed": None,
        "model_source": None,
        "training_episode": None,
        "evaluation_replicate_id": 0,
        "experiment_name": "always_call_vs_rule_based",
        "agent_name": "always_call",
    }
    pd.DataFrame([final_row, baseline_row]).to_csv(csv_path, index=False)

    result = calculate_final_model_metrics(str(csv_path))

    assert len(result) == 1
    assert result.iloc[0]["agent_name"] == "adaptive_mc"


def _game_row(*, seed, game_id, profit_bb):
    return {
        "training_run": "state_v2",
        "model_seed": seed,
        "model_source": "final",
        "training_episode": 5000,
        "checkpoint_episode": None,
        "experiment_id": f"seed_{seed}_episodes_5000",
        "experiment_name": "adaptive_mc_vs_calling",
        "game_id": game_id,
        "agent_name": "adaptive_mc",
        "opponent_name": "calling",
        "final_stack": 200,
        "initial_stack": 200,
        "profit": 0,
        "profit_bb": profit_bb,
        "hands_played": 20,
        "won_game": 1,
        "busted": 0,
        "ended_by_bust": 0,
        "ended_by_round_limit": 1,
        "classified_decisions": 0,
        "correct_classifications": 0,
        "incorrect_classifications": 0,
        "unknown_classifications": 0,
        "other_classifications": 0,
        "classifier_accuracy": 0.0,
        "classifier_coverage": 0.0,
        "policy_switches": 0,
        "policy_decisions": 0,
        "unseen_state_decisions": 0,
        "untried_action_selections": 0,
        "unseen_state_decision_rate": 0.0,
        "untried_action_selection_rate": 0.0,
        "first_classification_hand": 1,
        "first_correct_classification_hand": 1,
        "first_classification_action_count": 1,
        "first_correct_classification_action_count": 1,
        "final_predicted_type": "calling",
    }


def _metrics_for_seed_means(tmp_path, seed_means, games_per_seed=20):
    """One CSV where each seed is internally consistent but seeds differ."""
    rows = []
    game_id = 0
    for seed, mean in enumerate(seed_means, start=1):
        for index in range(games_per_seed):
            offset = 0.5 if index % 2 else -0.5
            rows.append(
                _game_row(seed=seed, game_id=game_id, profit_bb=mean + offset)
            )
            game_id += 1

    csv_path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return calculate_final_model_metrics(str(csv_path))


def test_game_level_interval_columns_are_named_as_descriptive(tmp_path):
    """The prefix is the whole point: these sit beside the seed-level columns."""
    metrics = _metrics_for_seed_means(tmp_path, [4.0])

    for column in GAME_LEVEL_COLUMNS:
        assert column in metrics.columns


def test_the_old_ambiguous_column_names_are_gone(tmp_path):
    metrics = _metrics_for_seed_means(tmp_path, [4.0])

    for column in ("standard_error", "ci_95_lower", "ci_95_upper"):
        assert column not in metrics.columns


def test_game_level_and_seed_level_column_names_do_not_collide():
    """The guard that makes the distinction permanent.

    Both layers report a standard error and a 95% interval. If a future column
    name lands in both namespaces, a reader cannot tell which layer a number
    came from - which is the confusion this PR exists to remove.
    """
    game_level = set(GAME_LEVEL_COLUMNS)
    seed_level = set(SEED_STATISTIC_COLUMNS)

    assert not game_level & seed_level

    for column in game_level:
        assert column.startswith("game_"), column
    for column in seed_level:
        assert "across_seeds" in column or "seed" in column, column


def test_game_level_interval_understates_uncertainty_across_seeds(tmp_path):
    """The substantive reason for the relabelling, not just the naming.

    Three seeds that disagree strongly, each internally consistent. Pooling the
    games treats 60 correlated observations as independent, so the interval
    collapses; the seed-level interval, which is the honest one, is far wider.
    """
    metrics = _metrics_for_seed_means(tmp_path, [2.0, 6.0, 10.0])
    aggregated = aggregate_across_seeds(metrics)

    row = aggregated.iloc[0]
    seed_width = float(row[SEED_CI_UPPER_COLUMN] - row[SEED_CI_LOWER_COLUMN])
    game_width = float(
        metrics[GAME_CI_UPPER].mean() - metrics[GAME_CI_LOWER].mean()
    )

    assert seed_width > game_width * 5, (
        f"seed-level width {seed_width:.3f} vs game-level {game_width:.3f}; "
        "the game-level interval should be dramatically narrower here"
    )


def test_game_level_interval_is_symmetric_around_the_mean(tmp_path):
    """Pins the normal approximation actually in use, so a silent change shows."""
    metrics = _metrics_for_seed_means(tmp_path, [4.0])
    row = metrics.iloc[0]

    below = row["mean_profit_bb"] - row[GAME_CI_LOWER]
    above = row[GAME_CI_UPPER] - row["mean_profit_bb"]

    assert below == pytest.approx(above)
    assert below == pytest.approx(1.96 * row[GAME_STANDARD_ERROR])


def test_every_game_level_column_has_a_description():
    """A relabelled column that no glossary explains is not documented."""
    for column in GAME_LEVEL_COLUMNS:
        assert column in METRIC_DESCRIPTIONS
        description = METRIC_DESCRIPTIONS[column].lower()
        assert "descriptive" in description or "not a basis" in description


def test_seed_level_descriptions_say_they_are_authoritative():
    for column in (SEED_CI_LOWER_COLUMN, SEED_CI_UPPER_COLUMN):
        assert "rest on" in METRIC_DESCRIPTIONS[column]


def test_the_constants_match_the_pinned_column_names():
    """Ties the literals above to the constants the source uses.

    Keeps the two from drifting without making the other tests depend on the
    constants, which would let a value change slip through unnoticed.
    """
    from src.evaluation.metrics.evaluation_metrics import (
        GAME_CI_LOWER_COLUMN,
        GAME_CI_UPPER_COLUMN,
        GAME_LEVEL_SPREAD_COLUMNS,
        GAME_STANDARD_ERROR_COLUMN,
    )

    assert GAME_STANDARD_ERROR_COLUMN == GAME_STANDARD_ERROR
    assert GAME_CI_LOWER_COLUMN == GAME_CI_LOWER
    assert GAME_CI_UPPER_COLUMN == GAME_CI_UPPER
    assert tuple(GAME_LEVEL_SPREAD_COLUMNS) == GAME_LEVEL_COLUMNS
