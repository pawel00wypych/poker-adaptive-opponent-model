from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.validation import (
    STATUS_FAIL,
    STATUS_PASS,
    ValidationCheckResult,
    ValidationReport,
    ValidationThresholds,
)
from src.experiments.validation import validate_evaluation_results as cli
from tests.evaluation.test_experiment_validation import (
    write_sample_final_model_csv,
)


def make_report(status: str) -> ValidationReport:
    return ValidationReport(
        input_path="results.csv",
        thresholds=ValidationThresholds(),
        checks=[
            ValidationCheckResult(
                check_name="sample",
                status=status,
                message="sample result",
                category="sample",
            )
        ],
    )


@pytest.mark.parametrize(
    ("status", "expected_exit_code", "expected_label"),
    [
        (STATUS_PASS, 0, "PASS"),
        (STATUS_FAIL, 1, "FAIL"),
    ],
)
def test_main_returns_exit_code_from_validation_result(
    monkeypatch,
    capsys,
    status,
    expected_exit_code,
    expected_label,
):
    args = SimpleNamespace(
        input_path="results.csv",
        output_dir="reports",
        format="both",
        validation_mode="training-opponent",
        algorithms=None,
        require_all_algorithms=False,
    )
    report = make_report(status)
    created_formats: list[str] = []
    validation_calls: list[dict] = []

    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        cli,
        "build_thresholds",
        lambda parsed_args: report.thresholds,
    )
    monkeypatch.setattr(
        cli,
        "validate_evaluation_results",
        lambda **kwargs: validation_calls.append(kwargs) or report,
    )
    monkeypatch.setattr(
        cli,
        "write_validation_markdown_report",
        lambda validation_report, output_dir: (
            created_formats.append("markdown")
            or Path(output_dir) / "experiment_validation.md"
        ),
    )
    monkeypatch.setattr(
        cli,
        "write_validation_json_report",
        lambda validation_report, output_dir: (
            created_formats.append("json")
            or Path(output_dir) / "experiment_validation.json"
        ),
    )

    exit_code = cli.main()

    assert exit_code == expected_exit_code
    assert created_formats == ["markdown", "json"]
    assert "training_episode" not in validation_calls[0]
    assert f"Validation status: {expected_label}" in capsys.readouterr().out


def test_module_returns_nonzero_exit_code_and_keeps_failure_report(tmp_path):
    input_path = tmp_path / "training_episode_results.csv"
    output_dir = tmp_path / "reports"
    write_sample_final_model_csv(input_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments.validation.validate_evaluation_results",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
            "--min-seeds-per-matchup",
            "3",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    report_path = output_dir / "experiment_validation.json"
    assert completed.returncode == 1
    assert "Validation status: FAIL" in completed.stdout
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False
