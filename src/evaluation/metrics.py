import numpy as np
import pandas as pd


def calculate_bb_per_100(
    results_csv_path: str,
) -> pd.DataFrame:
    df = pd.read_csv(
        results_csv_path
    )

    grouped = (
        df.groupby(
            [
                "experiment_name",
                "agent_name",
            ]
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
            mean_hands_played=(
                "hands_played",
                "mean",
            ),
            min_hands_played=(
                "hands_played",
                "min",
            ),
            max_hands_played=(
                "hands_played",
                "max",
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
        )
        .reset_index()
    )

    grouped["bb_per_100"] = (
        grouped["total_profit_bb"]
        / grouped["total_hands"]
        * 100
    )

    grouped["standard_error"] = (
        grouped["std_profit_bb"]
        / np.sqrt(grouped["games"])
    )

    grouped["ci_95_lower"] = (
        grouped["mean_profit_bb"]
        - 1.96
        * grouped["standard_error"]
    )

    grouped["ci_95_upper"] = (
        grouped["mean_profit_bb"]
        + 1.96
        * grouped["standard_error"]
    )

    total_evaluated = (
        grouped[
            "total_correct_classifications"
        ]
        + grouped[
            "total_incorrect_classifications"
        ]
    )

    grouped["global_classifier_accuracy"] = (
        np.where(
            total_evaluated > 0,
            grouped[
                "total_correct_classifications"
            ]
            / total_evaluated,
            0.0,
        )
    )

    total_predictions = (
        grouped[
            "total_classified_decisions"
        ]
        + grouped[
            "total_unknown_classifications"
        ]
    )

    grouped["global_classifier_coverage"] = (
        np.where(
            total_predictions > 0,
            grouped[
                "total_classified_decisions"
            ]
            / total_predictions,
            0.0,
        )
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