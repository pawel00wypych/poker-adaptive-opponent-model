import argparse

from src.evaluation.constants import SUPPORTED_TESTED_AGENTS
from src.evaluation.metrics.evaluation_metrics import (
    calculate_final_model_metrics,
)
from src.poker.constants import TRAINING_OPPONENT_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Show aggregated final-model training-opponent results.")
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
        choices=sorted(SUPPORTED_TESTED_AGENTS),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = calculate_final_model_metrics(args.input_path)

    if args.opponent is not None:
        df = df[df["opponent_name"] == args.opponent]

    if args.agent is not None:
        df = df[df["agent_name"] == args.agent]

    sort_columns = [
        "opponent_name",
        "agent_name",
        "training_episode",
        "model_seed",
    ]

    df = df.sort_values(sort_columns)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
