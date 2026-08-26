from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.reporting.experiment_summary import (
    dataframe_records_with_missing_as_none,
    write_dataframe_csv,
)
from src.evaluation.reporting.html_utils import write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a generic matchup overview for an evaluation CSV that "
            "does not have a dedicated scientific report generator."
        )
    )
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--title", required=True, type=str)
    return parser.parse_args(argv)


def build_evaluation_overview(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "agent_name",
        "opponent_name",
        "profit_bb",
        "hands_played",
        "won_game",
        "busted",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"Cannot create evaluation overview without columns: {missing}."
        )

    working = data.copy()
    for column in ("profit_bb", "hands_played", "won_game", "busted"):
        working[column] = pd.to_numeric(working[column], errors="raise")

    group_columns = ["agent_name", "opponent_name"]
    overview = (
        working.groupby(group_columns, dropna=False)
        .agg(
            games=("profit_bb", "size"),
            total_hands=("hands_played", "sum"),
            total_profit_bb=("profit_bb", "sum"),
            mean_profit_bb=("profit_bb", "mean"),
            win_rate=("won_game", "mean"),
            bust_rate=("busted", "mean"),
        )
        .reset_index()
    )
    overview["bb_per_100"] = np.where(
        overview["total_hands"] > 0,
        overview["total_profit_bb"] / overview["total_hands"] * 100.0,
        np.nan,
    )
    overview["win_rate"] *= 100.0
    overview["bust_rate"] *= 100.0

    unit_column = None
    for candidate in ("model_seed", "evaluation_replicate_id"):
        if candidate in working.columns and working[candidate].notna().any():
            unit_column = candidate
            break
    if unit_column is not None:
        per_unit = (
            working.dropna(subset=[unit_column])
            .groupby([*group_columns, unit_column], dropna=False)["profit_bb"]
            .mean()
            .reset_index(name="unit_mean_profit_bb")
        )
        uncertainty = (
            per_unit.groupby(group_columns, dropna=False)
            .agg(
                independent_units=(unit_column, "nunique"),
                mean_profit_bb_std_across_units=(
                    "unit_mean_profit_bb",
                    "std",
                ),
            )
            .reset_index()
        )
        overview = overview.merge(
            uncertainty,
            on=group_columns,
            how="left",
        )
    else:
        overview["independent_units"] = pd.NA
        overview["mean_profit_bb_std_across_units"] = pd.NA

    columns = [
        "agent_name",
        "opponent_name",
        "games",
        "independent_units",
        "mean_profit_bb",
        "mean_profit_bb_std_across_units",
        "bb_per_100",
        "win_rate",
        "bust_rate",
        "total_hands",
    ]
    result = overview[columns].sort_values(
        ["opponent_name", "mean_profit_bb", "agent_name"],
        ascending=[True, False, True],
    )
    for column in result.select_dtypes(include="number").columns:
        if column not in {"games", "independent_units", "total_hands"}:
            result[column] = result[column].round(3)
    return result.reset_index(drop=True)


def write_overview_outputs(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    title: str,
) -> list[Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = build_evaluation_overview(pd.read_csv(input_path))

    csv_path = write_dataframe_csv(
        overview,
        output_dir / "evaluation_overview.csv",
    )
    markdown_path = output_dir / "evaluation_overview.md"
    write_text(
        markdown_path,
        "\n".join(
            [
                f"# {title}",
                "",
                f"- **Input:** `{input_path}`",
                f"- **Raw games:** `{int(overview['games'].sum())}`",
                "",
                "## Matchup overview",
                "",
                overview.to_markdown(index=False),
                "",
            ]
        ),
    )
    json_path = output_dir / "evaluation_overview.json"
    write_text(
        json_path,
        json.dumps(
            {
                "title": title,
                "input_path": str(input_path),
                "rows": dataframe_records_with_missing_as_none(overview),
            },
            indent=2,
            allow_nan=False,
        ),
    )
    return [csv_path, markdown_path, json_path]


def main() -> None:
    args = parse_args()
    created = write_overview_outputs(
        input_path=args.input_path,
        output_dir=args.output_dir,
        title=args.title,
    )
    print("Created evaluation overview files:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
