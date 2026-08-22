from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.algorithm_metadata import (
    ADAPTIVE_AGENT_TO_ALGORITHM,
    ALGORITHM_ORDER,
)
from src.players.constants import GENERALIZATION_OPPONENT_TO_BASE_TYPE
from src.poker.constants import (
    OPPONENT_TYPE_UNKNOWN,
    TRAINING_OPPONENT_TYPES,
)

CLASSIFIER_COUNTER_COLUMNS = (
    "classified_decisions",
    "correct_classifications",
    "incorrect_classifications",
    "unknown_classifications",
    "other_classifications",
)

CLASSIFIER_QUALITY_REQUIRED_COLUMNS = {
    "training_run",
    "model_source",
    "model_seed",
    "training_episode",
    "agent_name",
    "opponent_name",
    "final_predicted_type",
    *CLASSIFIER_COUNTER_COLUMNS,
}

CLASSIFIER_QUALITY_SUMMARY_COLUMNS = [
    "training_run",
    "training_episode",
    "algorithm_name",
    "agent_name",
    "opponent_name",
    "opponent_family",
    "reference_type_available",
    "seeds",
    "games",
    "total_classified_decisions",
    "total_correct_classifications",
    "total_incorrect_classifications",
    "total_unknown_classifications",
    "total_other_classifications",
    "classification_opportunities",
    "global_classifier_accuracy",
    "classifier_coverage",
    "unknown_rate",
    "other_rate",
    "final_known_predictions",
    "final_unknown_predictions",
    "final_prediction_unknown_rate",
    "final_correct_predictions",
    "final_prediction_accuracy",
    "final_known_prediction_accuracy",
    "unexpected_final_predictions",
]

CONFUSION_MATRIX_COLUMNS = [
    "training_run",
    "training_episode",
    "algorithm_name",
    "agent_name",
    "actual_opponent_type",
    "predicted_opponent_type",
    "final_prediction_count",
    "actual_type_total",
    "row_percentage",
]

PREDICTED_TYPE_ORDER = (
    *TRAINING_OPPONENT_TYPES,
    OPPONENT_TYPE_UNKNOWN,
)


def _normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def opponent_family_for_name(opponent_name: object) -> str:
    normalized = _normalize_label(opponent_name)
    return GENERALIZATION_OPPONENT_TO_BASE_TYPE.get(
        normalized,
        normalized,
    )


def normalize_final_predicted_type(predicted_type: object) -> str:
    normalized = _normalize_label(predicted_type)
    return normalized or OPPONENT_TYPE_UNKNOWN


def calculate_classifier_summary(
    results_csv_path: str,
) -> pd.DataFrame:
    """Return the legacy per-opponent classifier summary."""
    df = pd.read_csv(results_csv_path)

    adaptive_df = df[df["agent_name"].isin(ADAPTIVE_AGENT_TO_ALGORITHM)].copy()

    adaptive_df["algorithm_name"] = adaptive_df["agent_name"].map(
        ADAPTIVE_AGENT_TO_ALGORITHM
    )
    adaptive_df["opponent_family"] = adaptive_df["opponent_name"].map(
        opponent_family_for_name
    )
    adaptive_df["final_prediction_correct"] = (
        adaptive_df["final_predicted_type"].map(normalize_final_predicted_type)
        == adaptive_df["opponent_family"]
    )

    summary = (
        adaptive_df.groupby(
            [
                "algorithm_name",
                "agent_name",
                "opponent_name",
                "opponent_family",
            ],
            sort=False,
        )
        .agg(
            games=("game_id", "count"),
            final_prediction_accuracy=(
                "final_prediction_correct",
                "mean",
            ),
            mean_classifier_accuracy=(
                "classifier_accuracy",
                "mean",
            ),
            mean_classifier_coverage=(
                "classifier_coverage",
                "mean",
            ),
            mean_policy_switches=(
                "policy_switches",
                "mean",
            ),
            mean_first_classification_hand=(
                "first_classification_hand",
                "mean",
            ),
            mean_first_correct_classification_hand=(
                "first_correct_classification_hand",
                "mean",
            ),
        )
        .reset_index()
    )

    percentage_columns = [
        "final_prediction_accuracy",
        "mean_classifier_accuracy",
        "mean_classifier_coverage",
    ]

    for column in percentage_columns:
        summary[column] *= 100

    return summary


def load_classifier_quality_rows(
    results_csv_path: str | Path,
) -> pd.DataFrame:
    """Load and normalize raw classifier output for adaptive agents."""
    data = pd.read_csv(results_csv_path)
    missing_columns = sorted(
        CLASSIFIER_QUALITY_REQUIRED_COLUMNS.difference(data.columns)
    )
    if missing_columns:
        raise ValueError(
            "Cannot create classifier quality report without columns: "
            f"{missing_columns}."
        )

    adaptive = data[data["agent_name"].isin(ADAPTIVE_AGENT_TO_ALGORITHM)].copy()
    if adaptive.empty:
        return adaptive.assign(
            algorithm_name=pd.Series(dtype="object"),
            opponent_family=pd.Series(dtype="object"),
            reference_type_available=pd.Series(dtype="bool"),
            normalized_final_predicted_type=pd.Series(dtype="object"),
            final_prediction_unknown=pd.Series(dtype="bool"),
            final_prediction_correct=pd.Series(dtype="bool"),
            unexpected_final_prediction=pd.Series(dtype="bool"),
        )

    invalid_sources = set(adaptive["model_source"].dropna()) - {"final"}
    if invalid_sources:
        raise ValueError(
            "Classifier quality input contains non-final model sources: "
            f"{sorted(invalid_sources)}. Use the learning-curve report for "
            "checkpoint analysis."
        )

    adaptive["training_episode"] = pd.to_numeric(
        adaptive["training_episode"],
        errors="raise",
    )
    if adaptive["training_episode"].isna().any():
        raise ValueError("Final-model classifier rows must contain training_episode.")
    if (
        "checkpoint_episode" in adaptive.columns
        and adaptive["checkpoint_episode"].notna().any()
    ):
        raise ValueError(
            "Final-model classifier rows must not contain checkpoint_episode."
        )
    for column in CLASSIFIER_COUNTER_COLUMNS:
        adaptive[column] = pd.to_numeric(
            adaptive[column],
            errors="raise",
        )
        if adaptive[column].isna().any() or (adaptive[column] < 0).any():
            raise ValueError(f"{column} must contain non-negative numeric values.")

    adaptive["algorithm_name"] = adaptive["agent_name"].map(ADAPTIVE_AGENT_TO_ALGORITHM)
    adaptive["opponent_family"] = adaptive["opponent_name"].map(
        opponent_family_for_name
    )
    adaptive["reference_type_available"] = adaptive["opponent_family"].isin(
        TRAINING_OPPONENT_TYPES
    )
    adaptive["normalized_final_predicted_type"] = adaptive["final_predicted_type"].map(
        normalize_final_predicted_type
    )
    adaptive["final_prediction_unknown"] = adaptive[
        "normalized_final_predicted_type"
    ].eq(OPPONENT_TYPE_UNKNOWN)
    adaptive["final_prediction_correct"] = adaptive[
        "reference_type_available"
    ] & adaptive["normalized_final_predicted_type"].eq(adaptive["opponent_family"])
    adaptive["unexpected_final_prediction"] = ~adaptive[
        "normalized_final_predicted_type"
    ].isin(PREDICTED_TYPE_ORDER)

    return adaptive


def select_final_classifier_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return all final-model rows without checkpoint-based selection."""
    if rows.empty:
        return rows.copy()

    return rows.sort_values(
        [
            "training_run",
            "training_episode",
            "algorithm_name",
            "opponent_name",
            "model_seed",
        ]
    ).reset_index(drop=True)


def _percentage(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    result = numerator.astype("float64").div(denominator.astype("float64")) * 100.0
    return result.where(denominator > 0)


def build_classifier_quality_summary(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate decision- and game-level classifier quality metrics."""
    if rows.empty:
        return pd.DataFrame(columns=CLASSIFIER_QUALITY_SUMMARY_COLUMNS)

    group_columns = [
        "training_run",
        "training_episode",
        "algorithm_name",
        "agent_name",
        "opponent_name",
        "opponent_family",
        "reference_type_available",
    ]
    summary = (
        rows.groupby(group_columns, dropna=False)
        .agg(
            seeds=("model_seed", "nunique"),
            games=("agent_name", "size"),
            total_classified_decisions=(
                "classified_decisions",
                "sum",
            ),
            total_correct_classifications=(
                "correct_classifications",
                "sum",
            ),
            total_incorrect_classifications=(
                "incorrect_classifications",
                "sum",
            ),
            total_unknown_classifications=(
                "unknown_classifications",
                "sum",
            ),
            total_other_classifications=(
                "other_classifications",
                "sum",
            ),
            final_known_predictions=(
                "final_prediction_unknown",
                lambda values: int((~values).sum()),
            ),
            final_unknown_predictions=(
                "final_prediction_unknown",
                "sum",
            ),
            final_correct_predictions=(
                "final_prediction_correct",
                "sum",
            ),
            unexpected_final_predictions=(
                "unexpected_final_prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    evaluated_classifications = (
        summary["total_correct_classifications"]
        + summary["total_incorrect_classifications"]
    )
    summary["classification_opportunities"] = (
        summary["total_classified_decisions"]
        + summary["total_unknown_classifications"]
        + summary["total_other_classifications"]
    )
    summary["global_classifier_accuracy"] = _percentage(
        summary["total_correct_classifications"],
        evaluated_classifications,
    ).where(summary["reference_type_available"])
    summary["classifier_coverage"] = _percentage(
        summary["total_classified_decisions"],
        summary["classification_opportunities"],
    )
    summary["unknown_rate"] = _percentage(
        summary["total_unknown_classifications"],
        summary["classification_opportunities"],
    )
    summary["other_rate"] = _percentage(
        summary["total_other_classifications"],
        summary["classification_opportunities"],
    )
    summary["final_prediction_unknown_rate"] = _percentage(
        summary["final_unknown_predictions"],
        summary["games"],
    )
    summary["final_prediction_accuracy"] = _percentage(
        summary["final_correct_predictions"],
        summary["games"],
    ).where(summary["reference_type_available"])
    summary["final_known_prediction_accuracy"] = _percentage(
        summary["final_correct_predictions"],
        summary["final_known_predictions"],
    ).where(summary["reference_type_available"])

    summary["_algorithm_order"] = summary["algorithm_name"].map(ALGORITHM_ORDER)
    summary = summary.sort_values(
        [
            "training_run",
            "training_episode",
            "_algorithm_order",
            "opponent_name",
        ]
    ).drop(columns="_algorithm_order")
    return summary[CLASSIFIER_QUALITY_SUMMARY_COLUMNS].reset_index(drop=True)


def build_classifier_confusion_matrix(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build a game-level matrix from each game's final prediction."""
    if rows.empty:
        return pd.DataFrame(columns=CONFUSION_MATRIX_COLUMNS)

    eligible = rows[rows["reference_type_available"]].copy()
    if eligible.empty:
        return pd.DataFrame(columns=CONFUSION_MATRIX_COLUMNS)

    context_columns = [
        "training_run",
        "training_episode",
        "algorithm_name",
        "agent_name",
    ]
    eligible = eligible.rename(
        columns={
            "opponent_family": "actual_opponent_type",
            "normalized_final_predicted_type": "predicted_opponent_type",
        }
    )
    observed = (
        eligible.groupby(
            [
                *context_columns,
                "actual_opponent_type",
                "predicted_opponent_type",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="final_prediction_count")
    )

    prediction_types = [
        *PREDICTED_TYPE_ORDER,
        *sorted(
            set(observed["predicted_opponent_type"]).difference(PREDICTED_TYPE_ORDER)
        ),
    ]
    complete_rows: list[dict[str, object]] = []
    for context_key, context in eligible.groupby(
        context_columns,
        dropna=False,
        sort=False,
    ):
        context_values = dict(zip(context_columns, context_key, strict=True))
        actual_types = [
            opponent_type
            for opponent_type in TRAINING_OPPONENT_TYPES
            if opponent_type in set(context["actual_opponent_type"])
        ]
        for actual_type in actual_types:
            for predicted_type in prediction_types:
                complete_rows.append(
                    {
                        **context_values,
                        "actual_opponent_type": actual_type,
                        "predicted_opponent_type": predicted_type,
                    }
                )

    complete = pd.DataFrame(complete_rows)
    matrix = complete.merge(
        observed,
        on=[
            *context_columns,
            "actual_opponent_type",
            "predicted_opponent_type",
        ],
        how="left",
    )
    matrix["final_prediction_count"] = (
        matrix["final_prediction_count"].fillna(0).astype("int64")
    )
    actual_group_columns = [*context_columns, "actual_opponent_type"]
    matrix["actual_type_total"] = matrix.groupby(
        actual_group_columns,
        dropna=False,
    )["final_prediction_count"].transform("sum")
    matrix["row_percentage"] = _percentage(
        matrix["final_prediction_count"],
        matrix["actual_type_total"],
    )

    matrix["_algorithm_order"] = matrix["algorithm_name"].map(ALGORITHM_ORDER)
    actual_order = {
        opponent_type: index
        for index, opponent_type in enumerate(TRAINING_OPPONENT_TYPES)
    }
    prediction_order = {
        opponent_type: index for index, opponent_type in enumerate(prediction_types)
    }
    matrix["_actual_order"] = matrix["actual_opponent_type"].map(actual_order)
    matrix["_prediction_order"] = matrix["predicted_opponent_type"].map(
        prediction_order
    )
    matrix = matrix.sort_values(
        [
            "training_run",
            "training_episode",
            "_algorithm_order",
            "_actual_order",
            "_prediction_order",
        ]
    ).drop(
        columns=[
            "_algorithm_order",
            "_actual_order",
            "_prediction_order",
        ]
    )
    return matrix[CONFUSION_MATRIX_COLUMNS].reset_index(drop=True)
