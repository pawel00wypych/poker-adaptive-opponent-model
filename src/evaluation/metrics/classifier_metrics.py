import pandas as pd

from src.evaluation.algorithm_metadata import ADAPTIVE_AGENT_TO_ALGORITHM
from src.players.constants import GENERALIZATION_OPPONENT_TO_BASE_TYPE


def calculate_classifier_summary(
    results_csv_path: str,
) -> pd.DataFrame:
    df = pd.read_csv(
        results_csv_path
    )

    adaptive_df = df[
        df["agent_name"].isin(ADAPTIVE_AGENT_TO_ALGORITHM)
    ].copy()

    adaptive_df["algorithm_name"] = adaptive_df["agent_name"].map(
        ADAPTIVE_AGENT_TO_ALGORITHM
    )
    adaptive_df["opponent_family"] = adaptive_df["opponent_name"].map(
        lambda opponent_name: GENERALIZATION_OPPONENT_TO_BASE_TYPE.get(
            opponent_name,
            opponent_name,
        )
    )
    adaptive_df["final_prediction_correct"] = (
        adaptive_df["final_predicted_type"]
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
