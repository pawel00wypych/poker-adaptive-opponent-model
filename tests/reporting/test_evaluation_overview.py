import json

import pandas as pd

from src.experiments.reporting.create_evaluation_overview import (
    build_evaluation_overview,
    parse_args,
    write_overview_outputs,
)


def sample_rows():
    return pd.DataFrame(
        [
            {
                "model_seed": 42,
                "agent_name": "adaptive_mc",
                "opponent_name": "tight_extreme",
                "profit_bb": 2.0,
                "hands_played": 20,
                "won_game": 1,
                "busted": 0,
            },
            {
                "model_seed": 123,
                "agent_name": "adaptive_mc",
                "opponent_name": "tight_extreme",
                "profit_bb": 4.0,
                "hands_played": 20,
                "won_game": 1,
                "busted": 0,
            },
        ]
    )


def test_overview_aggregates_matchup_and_independent_units():
    overview = build_evaluation_overview(sample_rows())
    row = overview.iloc[0]

    assert row["games"] == 2
    assert row["independent_units"] == 2
    assert row["mean_profit_bb"] == 3.0
    assert row["bb_per_100"] == 15.0
    assert row["win_rate"] == 100.0


def test_overview_writes_markdown_json_and_csv(tmp_path):
    input_path = tmp_path / "results.csv"
    sample_rows().to_csv(input_path, index=False)

    created = write_overview_outputs(
        input_path=input_path,
        output_dir=tmp_path / "report",
        title="Generalization Overview",
    )

    assert {path.name for path in created} == {
        "evaluation_overview.csv",
        "evaluation_overview.md",
        "evaluation_overview.json",
    }
    payload = json.loads(
        (tmp_path / "report" / "evaluation_overview.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["title"] == "Generalization Overview"
    assert payload["rows"][0]["mean_profit_bb"] == 3.0


def test_overview_cli_requires_title():
    args = parse_args(
        [
            "--input-path",
            "results.csv",
            "--output-dir",
            "reports",
            "--title",
            "Stress Overview",
        ]
    )

    assert args.title == "Stress Overview"
