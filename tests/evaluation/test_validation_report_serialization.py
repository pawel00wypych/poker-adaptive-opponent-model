import json

import pandas as pd

from src.evaluation.validation import (
    STATUS_PASS,
    ValidationCheckResult,
    ValidationReport,
    ValidationThresholds,
    write_validation_json_report,
)


def make_check_with_missing_values() -> ValidationCheckResult:
    return ValidationCheckResult(
        check_name="missing_values",
        status=STATUS_PASS,
        message="missing values are expected",
        category="serialization",
        observed_value=float("nan"),
        threshold=pd.NA,
        details={
            "missing_float": float("nan"),
            "nested": [pd.NA, {"missing_timestamp": pd.NaT}],
            "tuple_values": (1.0, float("nan")),
            "present_value": 2.5,
        },
    )


def test_validation_check_to_dict_converts_nested_missing_values_to_none():
    payload = make_check_with_missing_values().to_dict()

    assert payload["observed_value"] is None
    assert payload["threshold"] is None
    assert payload["details"] == {
        "missing_float": None,
        "nested": [None, {"missing_timestamp": None}],
        "tuple_values": (1.0, None),
        "present_value": 2.5,
    }


def test_json_report_serializes_all_missing_values_as_null(tmp_path):
    report = ValidationReport(
        input_path="results.csv",
        thresholds=ValidationThresholds(
            max_std_across_seeds_bb=float("nan"),
        ),
        checks=[make_check_with_missing_values()],
    )

    output_path = write_validation_json_report(report, tmp_path)
    raw_json = output_path.read_text(encoding="utf-8")
    payload = json.loads(raw_json)

    assert "NaN" not in raw_json
    assert '"observed_value": null' in raw_json
    assert payload["thresholds"]["max_std_across_seeds_bb"] is None
    assert payload["checks"][0]["observed_value"] is None
    assert payload["checks"][0]["threshold"] is None
    assert payload["checks"][0]["details"]["missing_float"] is None
    assert payload["checks"][0]["details"]["nested"] == [
        None,
        {"missing_timestamp": None},
    ]
