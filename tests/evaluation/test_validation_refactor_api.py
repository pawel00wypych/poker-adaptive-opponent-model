import json
from dataclasses import is_dataclass

import pandas as pd
import pytest

from src.evaluation import experiment_validation as legacy_validation
from src.evaluation import validation
from src.evaluation.validation import (
    STATUS_PASS,
    VALIDATION_MODE_GENERALIZATION,
    VALIDATION_MODE_HEAD_TO_HEAD,
    ValidationCheckResult,
    ValidationReport,
    ValidationThresholds,
    checkpoint_validation,
    validate_checkpoint_results,
    validation_checks_to_dataframe,
    write_validation_json_report,
    write_validation_markdown_report,
)


def make_validation_check(
    *,
    check_name="sample_check",
    status=STATUS_PASS,
    message="ok",
    category="sample_category",
):
    return ValidationCheckResult(
        check_name=check_name,
        status=status,
        message=message,
        category=category,
        algorithm_name="Monte Carlo",
        agent_name="adaptive_mc",
        opponent_name="calling",
        checkpoint_episode=1000,
        observed_value=1.0,
        threshold=0.0,
        details={"source": "test"},
    )


def make_validation_report():
    return ValidationReport(
        input_path="results/evaluation/sample.csv",
        thresholds=ValidationThresholds(
            min_classifier_accuracy=75.0,
            min_classifier_coverage=70.0,
        ),
        checks=[make_validation_check()],
    )


def test_validation_package_exports_public_api():
    expected_exports = [
        "ValidationThresholds",
        "ValidationCheckResult",
        "ValidationReport",
        "validate_checkpoint_results",
        "validate_expected_algorithms_present",
        "validate_minimum_seed_coverage",
        "validate_required_matchups_present",
        "validation_checks_to_dataframe",
        "render_validation_markdown",
        "write_validation_json_report",
        "write_validation_markdown_report",
        "AlgorithmValidationSpec",
        "ALGORITHM_VALIDATION_SPECS",
        "available_algorithm_specs",
    ]

    for exported_name in expected_exports:
        assert hasattr(validation, exported_name)

    assert set(expected_exports).issubset(validation.__all__)


def test_legacy_experiment_validation_wrapper_reexports_public_api():
    assert legacy_validation.__all__ == validation.__all__
    for exported_name in validation.__all__:
        assert getattr(legacy_validation, exported_name) is getattr(
            validation,
            exported_name,
        )


def test_validation_dataclasses_are_configurable_and_serializable():
    assert is_dataclass(ValidationThresholds)
    assert is_dataclass(ValidationCheckResult)
    assert is_dataclass(ValidationReport)

    thresholds = ValidationThresholds(
        min_classifier_accuracy=82.5,
        min_classifier_coverage=77.5,
        min_seeds_per_matchup=4,
    )
    check = make_validation_check()
    report = ValidationReport(
        input_path="input.csv",
        thresholds=thresholds,
        checks=[check],
        validation_mode=VALIDATION_MODE_HEAD_TO_HEAD,
    )

    payload = report.to_dict()

    assert thresholds.min_classifier_accuracy == 82.5
    assert thresholds.min_classifier_coverage == 77.5
    assert thresholds.min_seeds_per_matchup == 4
    assert payload["validation_mode"] == VALIDATION_MODE_HEAD_TO_HEAD
    assert payload["thresholds"]["min_classifier_accuracy"] == 82.5
    assert payload["thresholds"]["min_seeds_per_matchup"] == 4
    assert payload["checks"][0]["check_name"] == "sample_check"
    assert payload["status_counts"][STATUS_PASS] == 1


def test_validation_checks_to_dataframe_preserves_public_columns():
    check = make_validation_check()

    df = validation_checks_to_dataframe([check])

    assert list(df["check_name"]) == ["sample_check"]
    assert list(df["status"]) == [STATUS_PASS]
    assert list(df["category"]) == ["sample_category"]
    assert list(df["algorithm_name"]) == ["Monte Carlo"]
    assert "details" in df.columns


def test_validation_report_writers_create_markdown_and_json(tmp_path):
    report = make_validation_report()

    markdown_path = write_validation_markdown_report(report, tmp_path)
    json_path = write_validation_json_report(report, tmp_path)

    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert markdown_path.name == "experiment_validation.md"
    assert json_path.name == "experiment_validation.json"
    assert "Experiment validation report" in markdown
    assert "sample_check" in markdown
    assert payload["input_path"] == "results/evaluation/sample.csv"
    assert payload["checks"][0]["check_name"] == "sample_check"


def test_validate_checkpoint_results_rejects_unknown_validation_mode():
    with pytest.raises(ValueError, match="Unsupported validation_mode"):
        validate_checkpoint_results(
            "results/evaluation/sample.csv",
            validation_mode="unsupported-mode",
        )


def test_validate_checkpoint_results_dispatches_head_to_head_mode(
    monkeypatch,
    tmp_path,
):
    calls = []

    monkeypatch.setattr(
        checkpoint_validation,
        "load_checkpoint_report_data",
        lambda input_path: pd.DataFrame({"raw": [1]}),
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "aggregate_across_seeds",
        lambda metrics: pd.DataFrame({"aggregated": [1]}),
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "_add_mean_hands_played",
        lambda aggregated, metrics: aggregated,
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "_best_rows_by_agent_and_opponent",
        lambda aggregated: pd.DataFrame({"agent_name": ["adaptive_mc"]}),
    )

    def fake_head_to_head_validation(best_rows, thresholds, algorithm_specs=None):
        calls.append(("head_to_head", best_rows, thresholds, algorithm_specs))
        return [
            make_validation_check(
                check_name="head_to_head_delegate",
                category="head_to_head",
            )
        ]

    monkeypatch.setattr(
        checkpoint_validation,
        "validate_head_to_head_results_from_best_rows",
        fake_head_to_head_validation,
    )

    report = validate_checkpoint_results(
        tmp_path / "head_to_head.csv",
        validation_mode=VALIDATION_MODE_HEAD_TO_HEAD,
    )

    assert report.validation_mode == VALIDATION_MODE_HEAD_TO_HEAD
    assert [check.check_name for check in report.checks] == [
        "head_to_head_delegate"
    ]
    assert calls[0][0] == "head_to_head"
    assert list(calls[0][1]["agent_name"]) == ["adaptive_mc"]
    assert isinstance(calls[0][2], ValidationThresholds)
    assert calls[0][3][0].algorithm_name == "Monte Carlo"


def test_validate_checkpoint_results_dispatches_generalization_mode(
    monkeypatch,
    tmp_path,
):
    calls = []

    monkeypatch.setattr(
        checkpoint_validation,
        "load_checkpoint_report_data",
        lambda input_path: pd.DataFrame({"raw": [1]}),
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "aggregate_across_seeds",
        lambda metrics: pd.DataFrame({"aggregated": [1]}),
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "_add_mean_hands_played",
        lambda aggregated, metrics: aggregated,
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "_best_rows_by_agent_and_opponent",
        lambda aggregated: pd.DataFrame({"agent_name": ["adaptive_mc"]}),
    )

    def fake_generalization_validation(best_rows, thresholds, algorithm_specs=None):
        calls.append(("generalization", best_rows, thresholds, algorithm_specs))
        return [
            make_validation_check(
                check_name="generalization_delegate",
                category="generalization",
            )
        ]

    monkeypatch.setattr(
        checkpoint_validation,
        "validate_generalization_results_from_best_rows",
        fake_generalization_validation,
    )

    report = validate_checkpoint_results(
        tmp_path / "generalization.csv",
        validation_mode=VALIDATION_MODE_GENERALIZATION,
    )

    assert report.validation_mode == VALIDATION_MODE_GENERALIZATION
    assert [check.check_name for check in report.checks] == [
        "generalization_delegate"
    ]
    assert calls[0][0] == "generalization"
    assert list(calls[0][1]["agent_name"]) == ["adaptive_mc"]
    assert isinstance(calls[0][2], ValidationThresholds)
    assert calls[0][3][0].algorithm_name == "Monte Carlo"
