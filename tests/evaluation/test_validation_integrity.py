import json

import pandas as pd

from src.evaluation.validation import (
    STATUS_FAIL,
    STATUS_PASS,
    CheckKind,
    ValidationThresholds,
    ValidationCheckResult,
    validate_evaluation_results,
)
from tests.evaluation.test_additional_validation_modes import (
    write_sample_baseline_sanity_csv,
)
from tests.evaluation.test_experiment_validation import (
    write_sample_final_model_csv,
    write_sample_head_to_head_csv,
)


def write_summary(path, *, csv_path, games=1, seeds=None):
    path.write_text(
        json.dumps(
            {
                "output_path": str(csv_path),
                "row_count": len(
                    csv_path.read_text(encoding="utf-8").splitlines()
                )
                - 1,
                "games": games,
                "seeds": seeds,
            }
        ),
        encoding="utf-8",
    )


def test_manifest_and_exact_protocol_requirements_pass(tmp_path):
    csv_path = tmp_path / "results.csv"
    summary_path = csv_path.with_suffix(".summary.json")
    write_sample_final_model_csv(csv_path)
    write_summary(
        summary_path,
        csv_path=csv_path,
        seeds=[42, 123],
    )

    report = validate_evaluation_results(
        csv_path,
        thresholds=ValidationThresholds(
            expected_model_seeds=(42, 123),
            expected_games_per_matchup=1,
            require_manifest=True,
        ),
    )

    assert report.technically_valid
    assert report.schema_version == 2
    assert report.technical_status == STATUS_PASS
    manifest_check = next(
        check for check in report.checks if check.check_id == "evaluation_manifest"
    )
    assert manifest_check.check_type == CheckKind.INTEGRITY
    assert manifest_check.status == STATUS_PASS


def test_wrong_game_count_is_an_integrity_failure(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_sample_final_model_csv(csv_path)

    report = validate_evaluation_results(
        csv_path,
        thresholds=ValidationThresholds(expected_games_per_matchup=2),
    )

    assert not report.technically_valid
    game_count_check = next(
        check for check in report.checks if check.check_id == "games_per_matchup"
    )
    assert game_count_check.check_type == CheckKind.INTEGRITY
    assert game_count_check.status == STATUS_FAIL


def test_wrong_seed_set_is_an_integrity_failure(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_sample_final_model_csv(csv_path)

    report = validate_evaluation_results(
        csv_path,
        thresholds=ValidationThresholds(
            expected_model_seeds=(42, 123, 456),
        ),
    )

    assert not report.technically_valid
    seed_check = next(
        check for check in report.checks if check.check_id == "expected_model_seeds"
    )
    assert seed_check.status == STATUS_FAIL
    assert seed_check.details["mismatches"]


def test_split_threshold_views_preserve_compatibility():
    thresholds = ValidationThresholds(
        min_seeds_per_matchup=5,
        expected_games_per_matchup=500,
        min_classifier_accuracy=82.0,
    )

    assert thresholds.integrity_requirements.min_seeds_per_matchup == 5
    assert thresholds.integrity_requirements.expected_games_per_matchup == 500
    assert thresholds.diagnostic_thresholds.min_classifier_accuracy == 82.0


def test_mixed_head_to_head_rows_validate_model_and_replicate_parts(tmp_path):
    learned_path = tmp_path / "learned.csv"
    baseline_path = tmp_path / "baseline.csv"
    mixed_path = tmp_path / "mixed.csv"
    write_sample_head_to_head_csv(learned_path)
    write_sample_baseline_sanity_csv(baseline_path)
    learned = pd.read_csv(learned_path)
    baseline = pd.read_csv(baseline_path)
    baseline["model_source"] = None
    baseline["training_episode"] = None
    baseline["model_seed"] = None
    pd.concat([learned, baseline], ignore_index=True).to_csv(mixed_path, index=False)

    report = validate_evaluation_results(
        mixed_path,
        validation_mode="head-to-head",
    )

    assert report.technically_valid
    assert report.training_episode == 2000


def test_mixed_head_to_head_requires_minimum_replicate_coverage(tmp_path):
    learned_path = tmp_path / "learned.csv"
    baseline_path = tmp_path / "baseline.csv"
    mixed_path = tmp_path / "mixed.csv"
    write_sample_head_to_head_csv(learned_path)
    write_sample_baseline_sanity_csv(baseline_path)
    learned = pd.read_csv(learned_path)
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["evaluation_replicate_id"] == 0].copy()
    baseline["model_source"] = None
    baseline["training_episode"] = None
    baseline["model_seed"] = None
    pd.concat([learned, baseline], ignore_index=True).to_csv(mixed_path, index=False)

    report = validate_evaluation_results(
        mixed_path,
        validation_mode="head-to-head",
        thresholds=ValidationThresholds(
            min_evaluation_replicates_per_matchup=2,
        ),
    )

    assert not report.technically_valid
    coverage = next(
        check
        for check in report.checks
        if check.check_id == "expected_evaluation_replicates"
    )
    assert coverage.status == STATUS_FAIL


def test_mixed_head_to_head_rejects_rows_with_both_metadata_domains(tmp_path):
    csv_path = tmp_path / "mixed.csv"
    write_sample_head_to_head_csv(csv_path)
    rows = pd.read_csv(csv_path)
    rows.loc[0, "evaluation_replicate_id"] = 0
    rows.to_csv(csv_path, index=False)

    report = validate_evaluation_results(
        csv_path,
        validation_mode="head-to-head",
    )

    assert not report.technically_valid
    domain_check = next(
        check
        for check in report.checks
        if check.check_id == "metadata_domain_exclusivity"
    )
    assert domain_check.details["mixed_domain_rows"] == 1


def test_empty_required_manifest_is_an_integrity_failure(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_sample_final_model_csv(csv_path)
    csv_path.with_suffix(".summary.json").write_text("{}", encoding="utf-8")

    report = validate_evaluation_results(
        csv_path,
        thresholds=ValidationThresholds(require_manifest=True),
    )

    assert not report.technically_valid
    manifest_check = next(
        check for check in report.checks if check.check_id == "evaluation_manifest"
    )
    assert manifest_check.status == STATUS_FAIL
    assert manifest_check.details["errors"]


def test_malformed_manifest_seed_metadata_is_an_integrity_failure(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_sample_final_model_csv(csv_path)
    summary_path = csv_path.with_suffix(".summary.json")
    write_summary(summary_path, csv_path=csv_path, seeds="not-a-list")

    report = validate_evaluation_results(
        csv_path,
        thresholds=ValidationThresholds(require_manifest=True),
    )

    assert not report.technically_valid
    manifest_check = next(
        check for check in report.checks if check.check_id == "evaluation_manifest"
    )
    assert "seeds must be a list" in " ".join(manifest_check.details["errors"])


def test_fractional_model_seed_is_an_integrity_failure(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_sample_final_model_csv(csv_path)
    rows = pd.read_csv(csv_path)
    rows["model_seed"] = rows["model_seed"].astype(float)
    rows.loc[0, "model_seed"] = 42.5
    rows.to_csv(csv_path, index=False)

    report = validate_evaluation_results(csv_path)

    assert not report.technically_valid
    metadata_check = next(
        check for check in report.checks if check.check_id == "model_metadata_values"
    )
    assert metadata_check.status == STATUS_FAIL
    assert metadata_check.details["invalid_counts"]["model_seed"] == 1


def test_nan_metric_input_is_an_integrity_failure(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_sample_final_model_csv(csv_path)
    rows = pd.read_csv(csv_path)
    rows.loc[0, "classified_decisions"] = float("nan")
    rows.to_csv(csv_path, index=False)

    report = validate_evaluation_results(csv_path)

    assert not report.technically_valid
    finite_check = next(
        check for check in report.checks if check.check_id == "finite_numeric_values"
    )
    assert finite_check.details["invalid_counts"]["classified_decisions"] == 1


def test_validation_check_preserves_legacy_positional_arguments():
    check = ValidationCheckResult(
        "legacy",
        STATUS_PASS,
        "message",
        "category",
        "Monte Carlo",
    )

    assert check.algorithm_name == "Monte Carlo"
    assert check.check_type == CheckKind.DIAGNOSTIC
    assert check.to_dict()["check_type"] == "diagnostic"


def test_legacy_failure_check_infers_integrity():
    check = ValidationCheckResult(
        "legacy failure",
        STATUS_FAIL,
        "message",
        "category",
    )

    assert check.check_type == CheckKind.INTEGRITY
    assert check.blocking
