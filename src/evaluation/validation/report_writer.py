from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.evaluation.reporting.html_utils import write_text
from src.evaluation.reporting.training_opponent_report import display_agent_name
from src.evaluation.validation.models import (
    VALIDATION_STATUSES,
    CheckKind,
    ValidationCheckResult,
    ValidationReport,
)
from src.evaluation.validation.common import _format_float


def validation_checks_to_dataframe(
    checks: Iterable[ValidationCheckResult],
) -> pd.DataFrame:
    rows = [check.to_dict() for check in checks]

    if not rows:
        return pd.DataFrame(
            columns=[
                "check_name",
                "check_id",
                "check_type",
                "blocking",
                "hypothesis_id",
                "status",
                "category",
                "algorithm_name",
                "agent_name",
                "opponent_name",
                "training_episode",
                "observed_value",
                "threshold",
                "sample_size",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "message",
            ]
        )

    return pd.DataFrame(rows)


def _format_report_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df[
        [
            "status",
            "check_type",
            "blocking",
            "hypothesis_id",
            "category",
            "algorithm_name",
            "check_name",
            "agent_name",
            "opponent_name",
            "training_episode",
            "observed_value",
            "threshold",
            "sample_size",
            "standard_error",
            "ci_lower",
            "ci_upper",
            "message",
        ]
    ].copy()

    for column in [
        "observed_value",
        "threshold",
        "standard_error",
        "ci_lower",
        "ci_upper",
    ]:
        table[column] = table[column].map(_format_float)

    table["sample_size"] = table["sample_size"].map(
        lambda value: "n/a" if pd.isna(value) else str(int(value))
    )

    table["agent_name"] = table["agent_name"].map(
        lambda value: display_agent_name(value) if isinstance(value, str) else value
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
        "This report separates technical integrity checks from result diagnostics.",
        "",
        "## Input",
        "",
        f"- **Evaluation file:** `{report.input_path}`",
        f"- **Validation mode:** `{report.validation_mode}`",
        (
            f"- **Final training episode:** `{report.training_episode}`"
            if report.training_episode is not None
            else "- **Final training episode:** `n/a`"
        ),
        (
            f"- **Model selection:** `{report.model_selection}`"
            if report.model_selection is not None
            else "- **Model selection:** `n/a`"
        ),
        f"- **Report schema:** `{report.schema_version}`",
        f"- **Technical status:** `{report.technical_status}`",
        f"- **Technically valid:** `{str(report.technically_valid).lower()}`",
        "",
        "## Summary",
        "",
        status_table.to_markdown(index=False),
        "",
        "### Integrity counts",
        "",
        pd.DataFrame(
            [
                {"status": status, "count": count}
                for status, count in report.check_type_counts(
                    CheckKind.INTEGRITY
                ).items()
            ]
        ).to_markdown(index=False),
        "",
        "### Diagnostic counts",
        "",
        pd.DataFrame(
            [
                {"status": status, "count": count}
                for status, count in report.check_type_counts(
                    CheckKind.DIAGNOSTIC
                ).items()
            ]
        ).to_markdown(index=False),
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
    ]

    if checks_df.empty:
        lines.extend(["## Technical integrity", "", "No checks were generated."])
    else:
        integrity = checks_df[
            checks_df["check_type"] == CheckKind.INTEGRITY.value
        ]
        diagnostics = checks_df[
            checks_df["check_type"] == CheckKind.DIAGNOSTIC.value
        ]
        lines.extend(["## Technical integrity", ""])
        lines.append(
            "No integrity checks were generated."
            if integrity.empty
            else _format_report_table(integrity).to_markdown(index=False)
        )
        lines.extend(["", "## Performance diagnostics", ""])
        lines.append(
            "No diagnostic checks were generated."
            if diagnostics.empty
            else _format_report_table(diagnostics).to_markdown(index=False)
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
        json.dumps(report.to_dict(), indent=2, allow_nan=False),
    )
    return output_path
