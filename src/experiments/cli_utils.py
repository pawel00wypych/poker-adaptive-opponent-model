import argparse


def parse_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a poker Monte Carlo agent.",
    )

    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable periodic training progress logs. "
            "Use --progress or --no-progress."
        ),
    )

    parser.add_argument(
        "--player-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable detailed logs produced by the trained player."
        ),
    )

    parser.add_argument(
        "--player-log-interval",
        type=int,
        default=1,
        help=(
            "Print player logs every N poker rounds. "
            "Default: 1, meaning every round."
        ),
    )

    parser.add_argument(
        "--engine-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable internal PyPokerEngine game logs.",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print training progress every N episodes.",
    )

    args = parser.parse_args()

    if args.log_interval <= 0:
        parser.error("--log-interval must be greater than zero")

    if args.player_log_interval <= 0:
        parser.error("--player-log-interval must be greater than zero")

    return args