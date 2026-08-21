from pathlib import Path

import pandas as pd

from src.evaluation.reporting.experiment_charts import (
    MEAN_CI_CHART_FILENAME,
    SEED_STABILITY_CHART_FILENAME,
    ExperimentChartConfig,
    add_cross_seed_confidence_interval,
    create_experiment_summary_charts,
    plot_mean_profit_confidence_interval,
    plot_seed_stability,
)


def make_summary_table():
    return pd.DataFrame(
        [
            {
                "training_run": "run",
                "model_source": "final",
                "training_episode": 2000,
                "opponent_name": "calling",
                "agent_name": "adaptive_mc",
                "seeds": 4,
                "mean_profit_bb": 18.0,
                "mean_profit_bb_std_across_seeds": 2.0,
                "win_rate": 95.0,
                "bust_rate": 2.0,
            },
            {
                "training_run": "run",
                "model_source": "final",
                "training_episode": 2000,
                "opponent_name": "calling",
                "agent_name": "rule_based",
                "seeds": 4,
                "mean_profit_bb": -1.0,
                "mean_profit_bb_std_across_seeds": 0.5,
                "win_rate": 40.0,
                "bust_rate": 10.0,
            },
            {
                "training_run": "run",
                "model_source": "final",
                "training_episode": 1000,
                "opponent_name": "calling",
                "agent_name": "adaptive_mc",
                "seeds": 4,
                "mean_profit_bb": 10.0,
                "mean_profit_bb_std_across_seeds": 3.0,
                "win_rate": 80.0,
                "bust_rate": 5.0,
            },
        ]
    )


def assert_png(path: Path):
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(b"\x89PNG")


def test_add_cross_seed_confidence_interval_uses_seed_std_error():
    result = add_cross_seed_confidence_interval(
        make_summary_table(),
        ExperimentChartConfig(ci_multiplier=2.0),
    )
    row = result[
        (result["training_episode"] == 2000) & (result["agent_name"] == "adaptive_mc")
    ].iloc[0]

    assert row["mean_profit_bb_standard_error_across_seeds"] == 1.0
    assert row["mean_profit_bb_ci_95_error"] == 2.0
    assert row["mean_profit_bb_ci_95_lower"] == 16.0
    assert row["mean_profit_bb_ci_95_upper"] == 20.0


def test_plot_mean_profit_confidence_interval_creates_png(tmp_path):
    output_path = tmp_path / "mean_ci.png"

    result = plot_mean_profit_confidence_interval(
        make_summary_table(),
        output_path,
    )

    assert result == output_path
    assert_png(output_path)


def test_plot_seed_stability_creates_png(tmp_path):
    output_path = tmp_path / "seed_stability.png"

    result = plot_seed_stability(
        make_summary_table(),
        output_path,
    )

    assert result == output_path
    assert_png(output_path)


def test_create_experiment_summary_charts_creates_expected_files(tmp_path):
    created_paths = create_experiment_summary_charts(
        make_summary_table(),
        tmp_path,
    )
    created_names = {path.name for path in created_paths}

    assert created_names == {
        MEAN_CI_CHART_FILENAME,
        SEED_STABILITY_CHART_FILENAME,
    }
    for path in created_paths:
        assert_png(path)
