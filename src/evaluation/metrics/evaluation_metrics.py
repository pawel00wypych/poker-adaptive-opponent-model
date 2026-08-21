import numpy as np
import pandas as pd


def calculate_grouped_evaluation_metrics(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Calculate per-group metrics shared by model and baseline evaluations."""

    missing_columns = sorted(set(group_columns).difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Cannot calculate evaluation metrics without grouping columns: "
            f"{missing_columns}."
        )

    grouped = (
        df.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            total_profit_bb=(
                "profit_bb",
                "sum",
            ),
            total_hands=(
                "hands_played",
                "sum",
            ),
            games=(
                "game_id",
                "count",
            ),
            mean_profit_bb=(
                "profit_bb",
                "mean",
            ),
            std_profit_bb=(
                "profit_bb",
                "std",
            ),
            win_rate=(
                "won_game",
                "mean",
            ),
            bust_rate=(
                "busted",
                "mean",
            ),
            ended_by_bust_rate=(
                "ended_by_bust",
                "mean",
            ),
            ended_by_round_limit_rate=(
                "ended_by_round_limit",
                "mean",
            ),
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
            mean_first_classification_action_count=(
                "first_classification_action_count",
                "mean",
            ),
            mean_first_correct_classification_action_count=(
                "first_correct_classification_action_count",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["bb_per_100"] = grouped["total_profit_bb"] / grouped["total_hands"] * 100

    grouped["standard_error"] = grouped["std_profit_bb"] / np.sqrt(grouped["games"])

    grouped["ci_95_lower"] = (
        grouped["mean_profit_bb"] - 1.96 * grouped["standard_error"]
    )

    grouped["ci_95_upper"] = (
        grouped["mean_profit_bb"] + 1.96 * grouped["standard_error"]
    )

    total_evaluated = (
        grouped["total_correct_classifications"]
        + grouped["total_incorrect_classifications"]
    )

    grouped["global_classifier_accuracy"] = np.where(
        total_evaluated > 0,
        grouped["total_correct_classifications"] / total_evaluated,
        0.0,
    )

    total_predictions = (
        grouped["total_classified_decisions"] + grouped["total_unknown_classifications"]
    )

    grouped["global_classifier_coverage"] = np.where(
        total_predictions > 0,
        grouped["total_classified_decisions"] / total_predictions,
        0.0,
    )

    percentage_columns = [
        "win_rate",
        "bust_rate",
        "ended_by_bust_rate",
        "ended_by_round_limit_rate",
        "mean_classifier_accuracy",
        "mean_classifier_coverage",
        "global_classifier_accuracy",
        "global_classifier_coverage",
    ]

    for column in percentage_columns:
        grouped[column] *= 100

    return grouped


def calculate_final_model_metrics(
    results_csv_path: str,
) -> pd.DataFrame:
    df = pd.read_csv(results_csv_path)

    if "model_source" not in df.columns:
        raise ValueError("Evaluation data is missing model_source.")
    invalid_sources = set(df["model_source"].dropna()) - {"final"}
    if invalid_sources:
        raise ValueError(
            "Final evaluation data contains non-final model sources: "
            f"{sorted(invalid_sources)}."
        )
    df = df[df["model_source"] == "final"].copy()
    if df.empty:
        raise ValueError("Evaluation data contains no final-model rows.")
    if "training_episode" not in df.columns:
        raise ValueError("Evaluation data is missing training_episode.")
    if df["training_episode"].isna().any():
        raise ValueError("Final model rows must contain training_episode metadata.")
    if "checkpoint_episode" in df.columns and df["checkpoint_episode"].notna().any():
        raise ValueError("Final evaluation data must not contain checkpoint episodes.")

    return calculate_grouped_evaluation_metrics(
        df,
        [
            "training_run",
            "model_seed",
            "training_episode",
            "experiment_name",
            "agent_name",
            "opponent_name",
        ],
    )
