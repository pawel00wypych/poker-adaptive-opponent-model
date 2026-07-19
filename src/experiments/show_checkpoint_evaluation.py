import argparse

from src.evaluation.checkpoint_metrics import (
    calculate_checkpoint_metrics,
)
from src.experiments.constants import TESTED_AGENTS
from src.poker.constants import TRAINING_OPPONENT_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show aggregated checkpoint evaluation results."
        )
    )

    parser.add_argument(
        "--input-path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        choices=[
            *TRAINING_OPPONENT_TYPES,
        ],
    )

    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=TESTED_AGENTS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = calculate_checkpoint_metrics(
        args.input_path
    )

    if args.opponent is not None:
        df = df[
            df["opponent_name"] == args.opponent
        ]

    if args.agent is not None:
        df = df[
            df["agent_name"] == args.agent
        ]

    sort_columns = [
        "opponent_name",
        "agent_name",
        "checkpoint_episode",
        "model_seed",
    ]

    df = df.sort_values(
        sort_columns
    )

    print(
        df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
