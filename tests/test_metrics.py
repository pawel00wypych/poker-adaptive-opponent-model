import pandas as pd
import pytest

from src.evaluation.metrics import calculate_bb_per_100


def test_calculate_bb_per_100(tmp_path):
    csv_path = tmp_path / "results.csv"

    df = pd.DataFrame(
        [
            {
                "experiment_name": "adaptive_vs_fish",
                "game_id": 0,
                "agent_name": "adaptive_mc",
                "final_stack": 120,
                "initial_stack": 100,
                "profit": 20,
                "profit_bb": 2,
                "hands_played": 100,
                "won_game": 1,
                "busted": 0,
            },
            {
                "experiment_name": "adaptive_vs_fish",
                "game_id": 1,
                "agent_name": "adaptive_mc",
                "final_stack": 80,
                "initial_stack": 100,
                "profit": -20,
                "profit_bb": -2,
                "hands_played": 100,
                "won_game": 0,
                "busted": 0,
            },
            {
                "experiment_name": "adaptive_vs_fish",
                "game_id": 2,
                "agent_name": "adaptive_mc",
                "final_stack": 130,
                "initial_stack": 100,
                "profit": 30,
                "profit_bb": 3,
                "hands_played": 100,
                "won_game": 1,
                "busted": 0,
            },
        ]
    )

    df.to_csv(csv_path, index=False)

    summary = calculate_bb_per_100(str(csv_path))

    assert len(summary) == 1

    row = summary.iloc[0]

    assert row["total_profit_bb"] == 3
    assert row["total_hands"] == 300
    assert row["games"] == 3
    assert row["mean_profit_bb"] == 1.0
    assert row["mean_hands_played"] == 100
    assert row["min_hands_played"] == 100
    assert row["max_hands_played"] == 100
    assert row["bb_per_100"] == 1.0

    assert row["win_rate"] == pytest.approx(200 / 3)
    assert row["bust_rate"] == pytest.approx(0.0)

    assert row["standard_error"] > 0
    assert row["ci_95_lower"] < row["mean_profit_bb"]
    assert row["ci_95_upper"] > row["mean_profit_bb"]