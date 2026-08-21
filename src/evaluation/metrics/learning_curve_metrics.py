import pandas as pd

from src.evaluation.metrics.evaluation_metrics import (
    calculate_grouped_evaluation_metrics,
)


def calculate_learning_curve_metrics(
    results_csv_path: str,
) -> pd.DataFrame:
    df = pd.read_csv(results_csv_path)

    if "model_source" not in df.columns:
        raise ValueError("Learning-curve data is missing model_source.")
    invalid_sources = set(df["model_source"].dropna()) - {"checkpoint"}
    if invalid_sources:
        raise ValueError(
            "Learning-curve data contains non-checkpoint model sources: "
            f"{sorted(invalid_sources)}."
        )
    if df["checkpoint_episode"].isna().any():
        raise ValueError(
            "Learning-curve rows must contain checkpoint_episode metadata."
        )
    if "training_episode" in df.columns and df["training_episode"].notna().any():
        raise ValueError(
            "Learning-curve data must not contain final training episodes."
        )

    return calculate_grouped_evaluation_metrics(
        df,
        [
            "training_run",
            "model_seed",
            "checkpoint_episode",
            "experiment_name",
            "agent_name",
            "opponent_name",
        ],
    )
