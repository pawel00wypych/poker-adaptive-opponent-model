import json

import pandas as pd
import pytest

from src.evaluation.metrics.classifier_metrics import (
    build_classifier_confusion_matrix,
    build_classifier_quality_summary,
    load_classifier_quality_rows,
    select_final_classifier_rows,
)
from src.evaluation.reporting.classifier_quality import (
    ClassifierQualityConfig,
    build_classifier_quality_report,
    write_classifier_quality_outputs,
)


def make_classifier_row(
    *,
    game_id,
    agent,
    opponent,
    prediction,
    training_episode=2000,
    seed=1,
    classified=0,
    correct=0,
    incorrect=0,
    unknown=0,
    other=0,
):
    return {
        "training_run": "run",
        "model_seed": seed,
        "model_source": "final",
        "training_episode": training_episode,
        "game_id": game_id,
        "agent_name": agent,
        "opponent_name": opponent,
        "final_predicted_type": prediction,
        "classified_decisions": classified,
        "correct_classifications": correct,
        "incorrect_classifications": incorrect,
        "unknown_classifications": unknown,
        "other_classifications": other,
    }


def write_classifier_results(path):
    rows = [
        make_classifier_row(
            game_id=1,
            agent="adaptive_mc",
            opponent="calling_extreme",
            prediction="calling",
            classified=8,
            correct=6,
            incorrect=2,
            unknown=2,
        ),
        make_classifier_row(
            game_id=2,
            agent="adaptive_mc",
            opponent="calling_extreme",
            prediction="",
            seed=2,
            classified=2,
            correct=1,
            incorrect=1,
            unknown=8,
        ),
        make_classifier_row(
            game_id=3,
            agent="adaptive_mc",
            opponent="tight",
            prediction="aggressive",
            classified=4,
            correct=3,
            incorrect=1,
            unknown=1,
        ),
        make_classifier_row(
            game_id=4,
            agent="adaptive_mc",
            opponent="aggressive_extreme",
            prediction="tight",
            classified=5,
            correct=4,
            incorrect=1,
        ),
        make_classifier_row(
            game_id=5,
            agent="adaptive_mc",
            opponent="always_raise",
            prediction="unknown",
            unknown=5,
        ),
        make_classifier_row(
            game_id=6,
            agent="adaptive_q_learning",
            opponent="calling_extreme",
            prediction="calling",
            classified=10,
            correct=10,
        ),
        make_classifier_row(
            game_id=7,
            agent="adaptive_sarsa",
            opponent="calling_extreme",
            prediction="calling",
        ),
        make_classifier_row(
            game_id=8,
            agent="adaptive_double_q_learning",
            opponent="calling_extreme",
            prediction="unknown",
            unknown=10,
        ),
        make_classifier_row(
            game_id=9,
            agent="rule_based",
            opponent="calling_extreme",
            prediction="",
        ),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _summary_row(summary, agent, opponent):
    return summary[
        (summary["agent_name"] == agent) & (summary["opponent_name"] == opponent)
    ].iloc[0]


def _confusion_cell(matrix, agent, actual, predicted):
    return matrix[
        (matrix["agent_name"] == agent)
        & (matrix["actual_opponent_type"] == actual)
        & (matrix["predicted_opponent_type"] == predicted)
    ].iloc[0]


def test_quality_summary_uses_all_algorithms_and_pooled_unknown_rate(
    tmp_path,
):
    input_path = tmp_path / "results.csv"
    write_classifier_results(input_path)

    rows = load_classifier_quality_rows(input_path)
    selected = select_final_classifier_rows(rows)
    summary = build_classifier_quality_summary(selected)

    assert set(summary["agent_name"]) == {
        "adaptive_mc",
        "adaptive_q_learning",
        "adaptive_sarsa",
        "adaptive_double_q_learning",
    }
    assert set(summary["algorithm_name"]) == {
        "Monte Carlo",
        "Q-learning",
        "SARSA",
        "Double Q-learning",
    }
    assert set(selected["training_episode"]) == {2000}

    calling = _summary_row(summary, "adaptive_mc", "calling_extreme")
    assert calling["opponent_family"] == "calling"
    assert bool(calling["reference_type_available"]) is True
    assert calling["seeds"] == 2
    assert calling["classification_opportunities"] == 20
    assert calling["global_classifier_accuracy"] == pytest.approx(70.0)
    assert calling["classifier_coverage"] == pytest.approx(50.0)
    assert calling["unknown_rate"] == pytest.approx(50.0)
    assert calling["final_prediction_accuracy"] == pytest.approx(50.0)
    assert calling["final_prediction_unknown_rate"] == pytest.approx(50.0)
    assert calling["final_known_prediction_accuracy"] == pytest.approx(100.0)

    unsupported = _summary_row(summary, "adaptive_mc", "always_raise")
    assert bool(unsupported["reference_type_available"]) is False
    assert pd.isna(unsupported["global_classifier_accuracy"])
    assert pd.isna(unsupported["final_prediction_accuracy"])
    assert unsupported["unknown_rate"] == pytest.approx(100.0)

    no_decisions = _summary_row(
        summary,
        "adaptive_sarsa",
        "calling_extreme",
    )
    assert pd.isna(no_decisions["unknown_rate"])
    assert pd.isna(no_decisions["global_classifier_accuracy"])


def test_confusion_matrix_maps_variants_and_counts_unknown_final_prediction(
    tmp_path,
):
    input_path = tmp_path / "results.csv"
    write_classifier_results(input_path)
    rows = select_final_classifier_rows(load_classifier_quality_rows(input_path))

    matrix = build_classifier_confusion_matrix(rows)

    calling_correct = _confusion_cell(
        matrix,
        "adaptive_mc",
        "calling",
        "calling",
    )
    calling_unknown = _confusion_cell(
        matrix,
        "adaptive_mc",
        "calling",
        "unknown",
    )
    tight_as_aggressive = _confusion_cell(
        matrix,
        "adaptive_mc",
        "tight",
        "aggressive",
    )
    aggressive_as_tight = _confusion_cell(
        matrix,
        "adaptive_mc",
        "aggressive",
        "tight",
    )

    assert calling_correct["final_prediction_count"] == 1
    assert calling_correct["actual_type_total"] == 2
    assert calling_correct["row_percentage"] == pytest.approx(50.0)
    assert calling_unknown["final_prediction_count"] == 1
    assert calling_unknown["row_percentage"] == pytest.approx(50.0)
    assert tight_as_aggressive["final_prediction_count"] == 1
    assert aggressive_as_tight["final_prediction_count"] == 1
    assert "always_raise" not in set(matrix["actual_opponent_type"])


def test_report_uses_all_final_model_rows_without_episode_selection(
    tmp_path,
):
    input_path = tmp_path / "results.csv"
    write_classifier_results(input_path)

    report, summary, matrix = build_classifier_quality_report(input_path)

    assert report.overview["final_training_episodes"] == [2000]
    assert set(summary["training_episode"]) == {2000}
    assert set(matrix["training_episode"]) == {2000}
    assert "model_selection" in report.methodology


def test_classifier_quality_outputs_are_complete_and_json_uses_null(
    tmp_path,
):
    input_path = tmp_path / "results.csv"
    output_dir = tmp_path / "classifier_quality"
    write_classifier_results(input_path)

    created = write_classifier_quality_outputs(
        input_path=input_path,
        output_dir=output_dir,
        report_format="all",
        export_latex=True,
    )

    assert {path.name for path in created} == {
        "classifier_quality.md",
        "classifier_quality.json",
        "classifier_quality_summary.csv",
        "classifier_confusion_matrix.csv",
        "classifier_quality_summary.tex",
        "classifier_confusion_matrix.tex",
    }

    json_text = (output_dir / "classifier_quality.json").read_text(encoding="utf-8")
    payload = json.loads(json_text)
    assert "NaN" not in json_text
    sarsa = next(
        row
        for row in payload["quality_summary"]
        if row["agent_name"] == "adaptive_sarsa"
    )
    assert sarsa["unknown_rate"] is None
    assert payload["overview"]["games_excluded_from_confusion_matrix"] == 1

    markdown = (output_dir / "classifier_quality.md").read_text(encoding="utf-8")
    assert "# Classifier quality report" in markdown
    assert "decision unknown rate" in markdown
    assert "## Final-prediction confusion matrices" in markdown
    assert "games (row percentage)" in markdown


def test_classifier_quality_config_and_input_validation(tmp_path):
    assert ClassifierQualityConfig().__dict__ == {}

    input_path = tmp_path / "missing_column.csv"
    pd.DataFrame(
        [
            {
                "training_run": "run",
                "model_seed": 1,
                "model_source": "final",
                "training_episode": 1000,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
            }
        ]
    ).to_csv(input_path, index=False)
    with pytest.raises(
        ValueError,
        match="Cannot create classifier quality report without columns",
    ):
        load_classifier_quality_rows(input_path)


def test_summary_separates_other_from_unknown_and_covered_decisions(tmp_path):
    input_path = tmp_path / "other_rate.csv"
    pd.DataFrame(
        [
            make_classifier_row(
                game_id=1,
                agent="adaptive_mc",
                opponent="calling",
                prediction="calling",
                classified=5,
                correct=5,
                unknown=3,
                other=2,
            )
        ]
    ).to_csv(input_path, index=False)

    rows = load_classifier_quality_rows(input_path)
    summary = build_classifier_quality_summary(select_final_classifier_rows(rows))
    row = summary.iloc[0]

    assert row["total_other_classifications"] == 2
    assert row["classification_opportunities"] == 10
    assert row["classifier_coverage"] == pytest.approx(50.0)
    assert row["unknown_rate"] == pytest.approx(30.0)
    assert row["other_rate"] == pytest.approx(20.0)
