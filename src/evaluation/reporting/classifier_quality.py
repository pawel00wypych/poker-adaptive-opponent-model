from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.evaluation.algorithm_metadata import ALGORITHM_ORDER
from src.evaluation.metrics.classifier_metrics import (
    PREDICTED_TYPE_ORDER,
    build_classifier_confusion_matrix,
    build_classifier_quality_summary,
    load_classifier_quality_rows,
    select_final_classifier_rows,
)
from src.evaluation.reporting.experiment_summary import (
    dataframe_records_with_missing_as_none,
    write_dataframe_csv,
    write_dataframe_latex,
)
from src.evaluation.reporting.html_utils import write_text
from src.evaluation.reporting.training_opponent_report import display_agent_name
from src.poker.constants import TRAINING_OPPONENT_TYPES


@dataclass(frozen=True)
class ClassifierQualityConfig:
    """Configuration reserved for final-model classifier reporting."""


@dataclass(frozen=True)
class ClassifierQualityReport:
    input_path: str
    config: ClassifierQualityConfig
    methodology: dict[str, str]
    overview: dict[str, object]
    main_findings: list[str]
    quality_summary: list[dict[str, object]]
    confusion_matrix: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "config": asdict(self.config),
            "methodology": self.methodology,
            "overview": self.overview,
            "main_findings": self.main_findings,
            "quality_summary": self.quality_summary,
            "confusion_matrix": self.confusion_matrix,
        }


def _safe_percentage(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator * 100.0


def build_classifier_quality_overview(
    rows: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
) -> dict[str, object]:
    if rows.empty:
        return {
            "training_runs": [],
            "final_training_episodes": [],
            "model_seeds": [],
            "algorithms": [],
            "adaptive_agents": [],
            "opponents": [],
            "adaptive_games": 0,
            "games_with_reference_type": 0,
            "games_excluded_from_confusion_matrix": 0,
            "classification_opportunities": 0,
            "unknown_classifications": 0,
            "overall_unknown_rate": None,
            "overall_final_prediction_unknown_rate": None,
            "confusion_matrix_cells": 0,
        }

    classification_opportunities = float(
        rows["classified_decisions"].sum() + rows["unknown_classifications"].sum()
    )
    unknown_classifications = float(rows["unknown_classifications"].sum())
    final_unknown_predictions = int(rows["final_prediction_unknown"].sum())
    reference_games = int(rows["reference_type_available"].sum())
    algorithms = sorted(
        (str(value) for value in rows["algorithm_name"].dropna().unique()),
        key=lambda value: ALGORITHM_ORDER.get(value, len(ALGORITHM_ORDER)),
    )

    return {
        "training_runs": sorted(
            str(value) for value in rows["training_run"].dropna().unique()
        ),
        "final_training_episodes": sorted(
            int(value) for value in rows["training_episode"].dropna().unique()
        ),
        "model_seeds": sorted(
            int(value) for value in rows["model_seed"].dropna().unique()
        ),
        "algorithms": algorithms,
        "adaptive_agents": sorted(
            str(value) for value in rows["agent_name"].dropna().unique()
        ),
        "opponents": sorted(
            str(value) for value in rows["opponent_name"].dropna().unique()
        ),
        "adaptive_games": len(rows),
        "games_with_reference_type": reference_games,
        "games_excluded_from_confusion_matrix": len(rows) - reference_games,
        "classification_opportunities": int(classification_opportunities),
        "unknown_classifications": int(unknown_classifications),
        "overall_unknown_rate": _safe_percentage(
            unknown_classifications,
            classification_opportunities,
        ),
        "overall_final_prediction_unknown_rate": _safe_percentage(
            final_unknown_predictions,
            len(rows),
        ),
        "confusion_matrix_cells": len(confusion_matrix),
    }


def generate_classifier_quality_findings(
    summary: pd.DataFrame,
    overview: dict[str, object],
) -> list[str]:
    if summary.empty:
        return ["No adaptive-agent classifier results were available."]

    findings: list[str] = []
    with_unknown_rate = summary.dropna(subset=["unknown_rate"])
    if with_unknown_rate.empty:
        findings.append(
            "Decision-level unknown rate could not be calculated because "
            "there were no classification opportunities."
        )
    else:
        highest_unknown = with_unknown_rate.sort_values(
            ["unknown_rate", "classification_opportunities"],
            ascending=[False, False],
        ).iloc[0]
        findings.append(
            "Highest decision-level unknown rate: "
            f"{highest_unknown['agent_name']} vs "
            f"{highest_unknown['opponent_name']} "
            f"({float(highest_unknown['unknown_rate']):.3f}%; "
            f"{int(highest_unknown['total_unknown_classifications'])}/"
            f"{int(highest_unknown['classification_opportunities'])} "
            "classification opportunities)."
        )

    with_reference = summary.dropna(subset=["final_prediction_accuracy"])
    if with_reference.empty:
        findings.append(
            "Final-prediction accuracy could not be calculated because no "
            "supported reference opponent types were present."
        )
    else:
        lowest_accuracy = with_reference.sort_values(
            ["final_prediction_accuracy", "games"],
            ascending=[True, False],
        ).iloc[0]
        findings.append(
            "Lowest final-prediction accuracy: "
            f"{lowest_accuracy['agent_name']} vs "
            f"{lowest_accuracy['opponent_name']} "
            f"({float(lowest_accuracy['final_prediction_accuracy']):.3f}%; "
            f"{int(lowest_accuracy['games'])} games)."
        )

    excluded = int(overview["games_excluded_from_confusion_matrix"])
    if excluded:
        findings.append(
            f"Excluded {excluded} game(s) from the confusion matrix because "
            "their opponent did not map to tight, aggressive, or calling. "
            "Their unknown rates remain in the quality summary."
        )

    unexpected = int(summary["unexpected_final_predictions"].sum())
    if unexpected:
        findings.append(
            f"Found {unexpected} final prediction(s) outside the supported "
            "tight, aggressive, calling, and unknown labels."
        )

    return findings


def build_classifier_quality_report(
    input_path: str | Path,
    config: ClassifierQualityConfig | None = None,
) -> tuple[ClassifierQualityReport, pd.DataFrame, pd.DataFrame]:
    config = config or ClassifierQualityConfig()
    raw_rows = load_classifier_quality_rows(input_path)
    selected_rows = select_final_classifier_rows(raw_rows)
    summary = build_classifier_quality_summary(selected_rows)
    confusion_matrix = build_classifier_confusion_matrix(selected_rows)
    overview = build_classifier_quality_overview(
        selected_rows,
        confusion_matrix,
    )
    findings = generate_classifier_quality_findings(summary, overview)
    methodology = {
        "model_selection": (
            "All rows come from final.pkl. training_episode records the "
            "completed training budget and is never used to select an "
            "intermediate checkpoint."
        ),
        "decision_unknown_rate": (
            "unknown_classifications / (classified_decisions + "
            "unknown_classifications), calculated from pooled raw counts."
        ),
        "confusion_matrix": (
            "Game-level matrix of actual opponent family versus each game's "
            "final_predicted_type. It is not a per-decision matrix."
        ),
        "reference_types": (
            "Generalization variants are mapped to tight, aggressive, or "
            "calling. Matchups without one of these reference types are "
            "excluded only from accuracy and confusion-matrix calculations."
        ),
    }

    report = ClassifierQualityReport(
        input_path=str(input_path),
        config=config,
        methodology=methodology,
        overview=overview,
        main_findings=findings,
        quality_summary=dataframe_records_with_missing_as_none(summary),
        confusion_matrix=dataframe_records_with_missing_as_none(confusion_matrix),
    )
    return report, summary, confusion_matrix


def _round_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    identifier_columns = {
        "training_episode",
        "seeds",
        "games",
        "total_classified_decisions",
        "total_correct_classifications",
        "total_incorrect_classifications",
        "total_unknown_classifications",
        "classification_opportunities",
        "final_known_predictions",
        "final_unknown_predictions",
        "final_correct_predictions",
        "unexpected_final_predictions",
        "final_prediction_count",
        "actual_type_total",
    }
    for column in result.select_dtypes(include="number").columns:
        if column not in identifier_columns:
            result[column] = result[column].round(3)
    return result


def _display_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = _round_numeric_columns(dataframe)
    if "agent_name" in result.columns:
        result["agent_name"] = result["agent_name"].map(display_agent_name)
    return result


def _overview_markdown(overview: dict[str, object]) -> str:
    rows = [{"field": key, "value": value} for key, value in overview.items()]
    return pd.DataFrame(rows).to_markdown(index=False)


def _ordered_labels(values: pd.Series, preferred: tuple[str, ...]) -> list[str]:
    available = {str(value) for value in values}
    return [
        *[value for value in preferred if value in available],
        *sorted(available.difference(preferred)),
    ]


def render_confusion_matrices_markdown(
    confusion_matrix: pd.DataFrame,
) -> str:
    if confusion_matrix.empty:
        return "No games with a supported reference opponent type were available."

    sections: list[str] = []
    context_columns = [
        "training_run",
        "training_episode",
        "algorithm_name",
        "agent_name",
    ]
    for context_key, group in confusion_matrix.groupby(
        context_columns,
        sort=False,
    ):
        training_run, training_episode, algorithm_name, agent_name = context_key
        display_name = display_agent_name(str(agent_name))
        sections.extend(
            [
                (
                    f"### {algorithm_name} — {display_name} "
                    f"({training_run}, final episode {int(training_episode)})"
                ),
                "",
            ]
        )
        formatted = group.copy()
        formatted["result"] = formatted.apply(
            lambda row: (
                f"{int(row['final_prediction_count'])} "
                f"({float(row['row_percentage']):.1f}%)"
            ),
            axis=1,
        )
        matrix = formatted.pivot(
            index="actual_opponent_type",
            columns="predicted_opponent_type",
            values="result",
        )
        matrix = matrix.reindex(
            index=_ordered_labels(
                group["actual_opponent_type"],
                TRAINING_OPPONENT_TYPES,
            ),
            columns=_ordered_labels(
                group["predicted_opponent_type"],
                PREDICTED_TYPE_ORDER,
            ),
        )
        matrix.index.name = "actual / predicted"
        sections.extend([matrix.reset_index().to_markdown(index=False), ""])

    return "\n".join(sections).rstrip()


def render_classifier_quality_markdown(
    report: ClassifierQualityReport,
    summary: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
) -> str:
    findings = "\n".join(
        f"{index + 1}. {finding}" for index, finding in enumerate(report.main_findings)
    )
    methodology = "\n".join(
        f"- **{name.replace('_', ' ')}:** {description}"
        for name, description in report.methodology.items()
    )

    return "\n".join(
        [
            "# Classifier quality report",
            "",
            (
                "This report covers every adaptive algorithm present in the "
                "evaluation output. Percentages use the 0-to-100 scale."
            ),
            "",
            "## Methodology",
            "",
            methodology,
            "",
            "## Overview",
            "",
            _overview_markdown(report.overview),
            "",
            "## Main findings",
            "",
            findings,
            "",
            "## Classifier quality summary",
            "",
            _display_table(summary).to_markdown(index=False),
            "",
            "## Final-prediction confusion matrices",
            "",
            (
                "Each cell is `games (row percentage)`. Unknown is a "
                "prediction column; actual rows use known base families."
            ),
            "",
            render_confusion_matrices_markdown(confusion_matrix),
            "",
        ]
    )


def write_classifier_quality_outputs(
    input_path: str | Path,
    output_dir: str | Path,
    config: ClassifierQualityConfig | None = None,
    report_format: str = "all",
    export_latex: bool = True,
) -> list[Path]:
    if report_format not in {"markdown", "json", "both", "all"}:
        raise ValueError(
            "Unsupported report_format. Expected one of: markdown, json, both, all."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report, summary, confusion_matrix = build_classifier_quality_report(
        input_path,
        config=config,
    )

    created_paths: list[Path] = []
    if report_format in {"markdown", "both", "all"}:
        markdown_path = output_dir / "classifier_quality.md"
        write_text(
            markdown_path,
            render_classifier_quality_markdown(
                report,
                summary,
                confusion_matrix,
            ),
        )
        created_paths.append(markdown_path)

    if report_format in {"json", "both", "all"}:
        json_path = output_dir / "classifier_quality.json"
        write_text(
            json_path,
            json.dumps(report.to_dict(), indent=2, allow_nan=False),
        )
        created_paths.append(json_path)

    csv_exports = [
        ("classifier_quality_summary.csv", summary),
        ("classifier_confusion_matrix.csv", confusion_matrix),
    ]
    for filename, dataframe in csv_exports:
        created_paths.append(write_dataframe_csv(dataframe, output_dir / filename))

    if export_latex:
        latex_exports = [
            ("classifier_quality_summary.tex", summary),
            ("classifier_confusion_matrix.tex", confusion_matrix),
        ]
        for filename, dataframe in latex_exports:
            created_paths.append(
                write_dataframe_latex(dataframe, output_dir / filename)
            )

    return created_paths
