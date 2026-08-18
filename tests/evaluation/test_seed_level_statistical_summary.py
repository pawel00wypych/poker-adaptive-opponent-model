import json

import pandas as pd
import pytest

from src.evaluation.metrics.seed_statistics import (
    SEED_CI_LOWER_COLUMN,
    SEED_CI_MARGIN_COLUMN,
    SEED_CI_UPPER_COLUMN,
    SEED_MAX_COLUMN,
    SEED_MIN_COLUMN,
    SEED_SPREAD_COLUMN,
    SEED_STANDARD_ERROR_COLUMN,
    add_seed_level_statistical_summary,
)
from src.evaluation.reporting.checkpoint_report import aggregate_across_seeds
from src.evaluation.reporting.experiment_summary import (
    build_experiment_summary,
    write_experiment_summary_outputs,
)
from tests.reporting.test_experiment_summary import make_game_row


def make_per_seed_metric(
    *,
    seed: int,
    games: int,
    mean_profit_bb: float,
) -> dict[str, object]:
    return {
        "training_run": "run",
        "agent_name": "adaptive_mc",
        "opponent_name": "calling",
        "checkpoint_episode": 1000,
        "model_seed": seed,
        "games": games,
        "mean_profit_bb": mean_profit_bb,
        "bb_per_100": mean_profit_bb * 10.0,
        "win_rate": 50.0,
        "bust_rate": 10.0,
        "global_classifier_accuracy": 90.0,
        "global_classifier_coverage": 90.0,
        "mean_policy_switches": 1.0,
    }


def test_seed_summary_uses_student_t_confidence_interval():
    summary = pd.DataFrame(
        [
            {
                "seeds": 4,
                "mean_profit_bb": 2.5,
                "mean_profit_bb_std_across_seeds": 1.2909944487358056,
                SEED_MIN_COLUMN: 1.0,
                SEED_MAX_COLUMN: 4.0,
            }
        ]
    )

    row = add_seed_level_statistical_summary(summary).iloc[0]

    assert row[SEED_STANDARD_ERROR_COLUMN] == pytest.approx(
        0.6454972243679028
    )
    assert row[SEED_CI_MARGIN_COLUMN] == pytest.approx(2.0542602567605206)
    assert row[SEED_CI_LOWER_COLUMN] == pytest.approx(0.4457397432394794)
    assert row[SEED_CI_UPPER_COLUMN] == pytest.approx(4.554260256760521)
    assert row[SEED_SPREAD_COLUMN] == 3.0


def test_seed_summary_keeps_uncertainty_missing_for_one_seed():
    summary = pd.DataFrame(
        [
            {
                "seeds": 1,
                "mean_profit_bb": 5.0,
                "mean_profit_bb_std_across_seeds": float("nan"),
                SEED_MIN_COLUMN: 5.0,
                SEED_MAX_COLUMN: 5.0,
            }
        ]
    )

    row = add_seed_level_statistical_summary(summary).iloc[0]

    assert pd.isna(row[SEED_STANDARD_ERROR_COLUMN])
    assert pd.isna(row[SEED_CI_MARGIN_COLUMN])
    assert pd.isna(row[SEED_CI_LOWER_COLUMN])
    assert pd.isna(row[SEED_CI_UPPER_COLUMN])
    assert row[SEED_SPREAD_COLUMN] == 0.0


def test_aggregate_across_seeds_equally_weights_each_training_seed():
    metrics = pd.DataFrame(
        [
            make_per_seed_metric(seed=42, games=1000, mean_profit_bb=0.0),
            make_per_seed_metric(seed=123, games=1, mean_profit_bb=10.0),
        ]
    )

    row = aggregate_across_seeds(metrics).iloc[0]

    assert row["seeds"] == 2
    assert row["games"] == 1001
    assert row["mean_profit_bb"] == 5.0
    assert row["mean_profit_bb_std_across_seeds"] == pytest.approx(
        7.0710678118654755
    )
    assert row[SEED_STANDARD_ERROR_COLUMN] == pytest.approx(5.0)
    assert row[SEED_CI_MARGIN_COLUMN] == pytest.approx(63.53102368087347)
    assert row[SEED_MIN_COLUMN] == 0.0
    assert row[SEED_MAX_COLUMN] == 10.0
    assert row[SEED_SPREAD_COLUMN] == 10.0


def test_experiment_summary_exports_seed_statistics_and_null_for_one_seed(
    tmp_path,
):
    csv_path = tmp_path / "one_seed.csv"
    output_dir = tmp_path / "summary"
    pd.DataFrame(
        [
            make_game_row(
                seed=42,
                checkpoint=1000,
                agent="adaptive_mc",
                opponent="calling",
                game_id=0,
                profit_bb=5.0,
            )
        ]
    ).to_csv(csv_path, index=False)

    report, ranking, _, _ = build_experiment_summary(csv_path)
    assert pd.isna(ranking.iloc[0][SEED_CI_LOWER_COLUMN])
    assert report.ranking[0][SEED_STANDARD_ERROR_COLUMN] is None
    assert report.ranking[0][SEED_CI_LOWER_COLUMN] is None

    write_experiment_summary_outputs(
        input_path=csv_path,
        output_dir=output_dir,
        report_format="json",
        export_latex=False,
        include_charts=False,
    )
    json_text = (output_dir / "experiment_summary.json").read_text(
        encoding="utf-8"
    )
    payload = json.loads(json_text)

    assert payload["ranking"][0][SEED_CI_LOWER_COLUMN] is None
    assert "NaN" not in json_text
