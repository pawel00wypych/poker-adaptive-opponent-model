import argparse

from src.evaluation.checkpoint_metrics import (
    calculate_checkpoint_metrics,
)


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
            "fish",
            "aggressive",
            "calling",
        ],
    )

    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=[
            "rule_based",
            "single_policy_mc",
            "adaptive_mc",
        ],
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