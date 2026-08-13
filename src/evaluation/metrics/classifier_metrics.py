import pandas as pd


def calculate_classifier_summary(
    results_csv_path: str,
) -> pd.DataFrame:
    df = pd.read_csv(
        results_csv_path
    )

    adaptive_df = df[
        df["agent_name"] == "adaptive_mc"
    ].copy()

    adaptive_df[
        "final_prediction_correct"
    ] = (
        adaptive_df["final_predicted_type"]
        == adaptive_df["opponent_name"]
    )

    summary = (
        adaptive_df.groupby(
            "opponent_name"
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