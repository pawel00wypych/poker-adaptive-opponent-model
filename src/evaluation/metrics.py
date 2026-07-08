import pandas as pd


def calculate_bb_per_100(results_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(results_csv_path)

    grouped = (
        df.groupby(["experiment_name", "agent_name"])
        .agg(
            total_profit_bb=("profit_bb", "sum"),
            total_hands=("hands_played", "sum"),
            games=("game_id", "count"),
            mean_profit_bb=("profit_bb", "mean"),
            std_profit_bb=("profit_bb", "std"),
            mean_hands_played=("hands_played", "mean"),
            min_hands_played=("hands_played", "min"),
            max_hands_played=("hands_played", "max"),
        )
        .reset_index()
    )

    grouped["bb_per_100"] = grouped["total_profit_bb"] / grouped["total_hands"] * 100

    return grouped