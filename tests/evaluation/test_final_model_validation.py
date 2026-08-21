import json

import pandas as pd
import pytest

from src.evaluation.validation import (
    render_validation_markdown,
    validate_evaluation_results,
    write_validation_json_report,
)
from src.experiments.validation.validate_evaluation_results import parse_args
from tests.evaluation.test_experiment_validation import (
    write_sample_final_model_csv,
)


def test_validation_reports_final_model_selection(tmp_path):
    csv_path = tmp_path / "final_models.csv"
    write_sample_final_model_csv(csv_path)

    report = validate_evaluation_results(csv_path)

    assert report.training_episode == 2000
    assert report.model_selection == "final"
    assert {
        check.training_episode
        for check in report.checks
        if check.training_episode is not None
    } == {2000}


def test_validation_rejects_multiple_training_episodes_as_mixed_input(tmp_path):
    csv_path = tmp_path / "mixed_final_models.csv"
    single_path = tmp_path / "single.csv"
    write_sample_final_model_csv(single_path)
    final_rows = pd.read_csv(single_path)
    earlier_rows = final_rows.copy()
    earlier_rows["training_episode"] = 1000
    pd.concat([earlier_rows, final_rows], ignore_index=True).to_csv(
        csv_path,
        index=False,
    )

    with pytest.raises(ValueError, match="exactly one training episode"):
        validate_evaluation_results(csv_path)


def test_validation_rejects_learning_curve_checkpoint_rows(tmp_path):
    csv_path = tmp_path / "checkpoint_rows.csv"
    final_path = tmp_path / "final.csv"
    write_sample_final_model_csv(final_path)
    rows = pd.read_csv(final_path)
    rows["model_source"] = "checkpoint"
    rows["checkpoint_episode"] = rows["training_episode"]
    rows["training_episode"] = None
    rows.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="non-final model sources"):
        validate_evaluation_results(csv_path)


def test_validation_report_serializes_final_model_selection(tmp_path):
    csv_path = tmp_path / "final_models.csv"
    write_sample_final_model_csv(csv_path)
    report = validate_evaluation_results(csv_path)

    markdown = render_validation_markdown(report)
    json_path = write_validation_json_report(report, tmp_path / "report")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "**Final training episode:** `2000`" in markdown
    assert "**Model selection:** `final`" in markdown
    assert payload["training_episode"] == 2000
    assert payload["model_selection"] == "final"


def test_validation_cli_does_not_accept_checkpoint_override():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input-path",
                "results/evaluation/results.csv",
                "--output-dir",
                "reports/validation",
                "--checkpoint-episode",
                "1500",
            ]
        )
