import argparse


def add_common_training_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic training progress logs.",
    )

    parser.add_argument(
        "--player-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable detailed player logs.",
    )

    parser.add_argument(
        "--player-log-interval",
        type=int,
        default=1,
        help="Print player logs every N rounds.",
    )

    parser.add_argument(
        "--engine-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable PyPokerEngine logs.",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print training progress every N episodes.",
    )


def validate_training_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    if args.log_interval <= 0:
        parser.error(
            "--log-interval must be greater than zero"
        )

    if args.player_log_interval <= 0:
        parser.error(
            "--player-log-interval must be greater than zero"
        )

    return args


def parse_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a general Monte Carlo poker agent.",
    )

    add_common_training_arguments(parser)

    args = parser.parse_args()

    return validate_training_args(
        parser,
        args,
    )


def parse_specialist_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Monte Carlo specialist against "
            "one opponent type."
        ),
    )

    parser.add_argument(
        "--opponent",
        required=True,
        choices=[
            "fish",
            "aggressive",
            "calling",
        ],
        help="Opponent type used during specialist training.",
    )

    add_common_training_arguments(parser)

    args = parser.parse_args()

    return validate_training_args(
        parser,
        args,
    )