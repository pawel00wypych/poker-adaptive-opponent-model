import json

import pandas as pd
import pytest

from src.evaluation.validation import (
    STATUS_WARNING,
    render_validation_markdown,
    validate_checkpoint_results,
    write_validation_json_report,
)
from src.experiments.validation.validate_checkpoint_evaluation import parse_args
from tests.evaluation.test_experiment_validation import (
    write_sample_checkpoint_csv,
)


def write_multiple_checkpoint_csv(path):
    latest_path = path.with_name("latest_only.csv")
    write_sample_checkpoint_csv(latest_path)
    latest = pd.read_csv(latest_path)

    earlier = latest.copy()
    earlier["checkpoint_episode"] = 1000

    earlier_adaptive_aggressive = (
        (earlier["agent_name"] == "adaptive_mc")
        & (earlier["opponent_name"] == "aggressive")
    )
    earlier.loc[earlier_adaptive_aggressive, "profit_bb"] = 100.0
    earlier.loc[earlier_adaptive_aggressive, "correct_classifications"] = 10
    earlier.loc[earlier_adaptive_aggressive, "incorrect_classifications"] = 0
    earlier.loc[earlier_adaptive_aggressive, "classifier_accuracy"] = 1.0

    latest_adaptive_aggressive = (
        (latest["agent_name"] == "adaptive_mc")
        & (latest["opponent_name"] == "aggressive")
    )
    latest.loc[latest_adaptive_aggressive, "correct_classifications"] = 4
    latest.loc[latest_adaptive_aggressive, "incorrect_classifications"] = 6
    latest.loc[latest_adaptive_aggressive, "classifier_accuracy"] = 0.4

    pd.concat([earlier, latest], ignore_index=True).to_csv(path, index=False)


def classifier_accuracy_check(report):
    return next(
        check
        for check in report.checks
        if check.check_name
        == "Monte Carlo: Adaptive classifier accuracy vs aggressive"
    )


def test_validation_uses_latest_checkpoint_instead_of_best_profit(tmp_path):
    csv_path = tmp_path / "multiple_checkpoints.csv"
    write_multiple_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(csv_path)
    check = classifier_accuracy_check(report)

    assert report.checkpoint_episode == 2000
    assert report.checkpoint_selection == "latest"
    assert check.checkpoint_episode == 2000
    assert check.status == STATUS_WARNING
    assert check.observed_value == pytest.approx(40.0)
    assert {
        item.checkpoint_episode
        for item in report.checks
        if item.checkpoint_episode is not None
    } == {2000}


def test_explicit_checkpoint_overrides_latest_selection(tmp_path):
    csv_path = tmp_path / "multiple_checkpoints.csv"
    write_multiple_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        checkpoint_episode=1000,
    )
    check = classifier_accuracy_check(report)

    assert report.checkpoint_episode == 1000
    assert report.checkpoint_selection == "explicit"
    assert check.checkpoint_episode == 1000
    assert check.observed_value == pytest.approx(100.0)


def test_validation_rejects_missing_explicit_checkpoint(tmp_path):
    csv_path = tmp_path / "multiple_checkpoints.csv"
    write_multiple_checkpoint_csv(csv_path)

    with pytest.raises(ValueError, match="Checkpoint 3000 is not present"):
        validate_checkpoint_results(
            csv_path,
            checkpoint_episode=3000,
        )


def test_validation_report_serializes_checkpoint_selection(tmp_path):
    csv_path = tmp_path / "multiple_checkpoints.csv"
    write_multiple_checkpoint_csv(csv_path)
    report = validate_checkpoint_results(csv_path)

    markdown = render_validation_markdown(report)
    json_path = write_validation_json_report(report, tmp_path / "report")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "**Selected checkpoint:** `2000`" in markdown
    assert "**Checkpoint selection:** `latest`" in markdown
    assert payload["checkpoint_episode"] == 2000
    assert payload["checkpoint_selection"] == "latest"


def test_validation_cli_accepts_explicit_checkpoint_override():
    args = parse_args(
        [
            "--input-path",
            "results/evaluation/results.csv",
            "--output-dir",
            "reports/validation",
            "--checkpoint-episode",
            "1500",
        ]
    )

    assert args.checkpoint_episode == 1500
