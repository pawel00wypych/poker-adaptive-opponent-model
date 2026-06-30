import pandas as pd

from src.evaluation.metrics import calculate_bb_per_100


def test_calculate_bb_per_100(tmp_path):
    csv_path = tmp_path / "results.csv"

    df = pd.DataFrame(
        [
            {
                "experiment_name": "adaptive_vs_fish",
                "game_id": 0,
                "agent_name": "adaptive_rl",
                "final_stack": 120,
                "initial_stack": 100,
                "profit": 20,
                "profit_bb": 2,
                "hands_played": 100,
            },
            {
                "experiment_name": "adaptive_vs_fish",
                "game_id": 1,
                "agent_name": "adaptive_rl",
                "final_stack": 80,
                "initial_stack": 100,
                "profit": -20,
                "profit_bb": -2,
                "hands_played": 100,
            },
            {
                "experiment_name": "adaptive_vs_fish",
                "game_id": 2,
                "agent_name": "adaptive_rl",
                "final_stack": 130,
                "initial_stack": 100,
                "profit": 30,
                "profit_bb": 3,
                "hands_played": 100,
            },
        ]
    )

    df.to_csv(csv_path, index=False)

    summary = calculate_bb_per_100(str(csv_path))

    assert len(summary) == 1
    assert summary.iloc[0]["total_profit_bb"] == 3
    assert summary.iloc[0]["total_hands"] == 300
    assert summary.iloc[0]["bb_per_100"] == 1.0