from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.evaluation.validation.common import (
    VALIDATION_STATUSES,
    ValidationCheckResult,
    ValidationReport,
    _format_float,
    display_agent_name,
    write_text,
)


def validation_checks_to_dataframe(
    checks: Iterable[ValidationCheckResult],
) -> pd.DataFrame:
    rows = [
        check.to_dict()
        for check in checks
    ]

    if not rows:
        return pd.DataFrame(
            columns=[
                "check_name",
                "status",
                "category",
                "algorithm_name",
                "agent_name",
                "opponent_name",
                "checkpoint_episode",
                "observed_value",
                "threshold",
                "message",
            ]
        )

    return pd.DataFrame(rows)

def _format_report_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df[
        [
            "status",
            "category",
            "algorithm_name",
            "check_name",
            "agent_name",
            "opponent_name",
            "checkpoint_episode",
            "observed_value",
            "threshold",
            "message",
        ]
    ].copy()

    for column in ["observed_value", "threshold"]:
        table[column] = table[column].map(_format_float)

    table["agent_name"] = table["agent_name"].map(
        lambda value: display_agent_name(value)
        if isinstance(value, str)
        else value
    )

    return table

def render_validation_markdown(report: ValidationReport) -> str:
    checks_df = validation_checks_to_dataframe(report.checks)
    counts = report.status_counts()
    status_table = pd.DataFrame(
        [
            {
                "status": status,
                "count": counts[status],
            }
            for status in VALIDATION_STATUSES
        ]
    )

    lines = [
        "# Experiment validation report",
        "",
        "This report runs automated sanity checks on evaluation "
        "results.",
        "",
        "## Input",
        "",
        f"- **Evaluation file:** `{report.input_path}`",
        f"- **Validation mode:** `{report.validation_mode}`",
        f"- **Overall status:** `{'PASS' if report.passed else 'FAIL'}`",
        "",
        "## Status summary",
        "",
        status_table.to_markdown(index=False),
        "",
        "## Thresholds",
        "",
        pd.DataFrame(
            [
                {
                    "threshold": key,
                    "value": value,
                }
                for key, value in asdict(report.thresholds).items()
            ]
        ).to_markdown(index=False),
        "",
        "## Checks",
        "",
    ]

    if checks_df.empty:
        lines.append("No checks were generated.")
    else:
        lines.append(
            _format_report_table(checks_df).to_markdown(index=False)
        )

    lines.append("")
    return "\n".join(lines)

def write_validation_markdown_report(
    report: ValidationReport,
    output_dir: str | Path,
    filename: str = "experiment_validation.md",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    write_text(
        output_path,
        render_validation_markdown(report),
    )
    return output_path

def write_validation_json_report(
    report: ValidationReport,
    output_dir: str | Path,
    filename: str = "experiment_validation.json",
) -> Path:
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    write_text(
        output_path,
        json.dumps(report.to_dict(), indent=2),
    )
    return output_path
