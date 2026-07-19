import argparse

from src.poker.constants import TRAINING_OPPONENT_TYPES
from src.training.constants import (
    SUPPORTED_ALPHA_MODES,
    SUPPORTED_EPSILON_SCHEDULES,
)


def add_common_training_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable periodic training progress logs."
        ),
    )

    parser.add_argument(
        "--player-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable detailed player logs."
        ),
    )

    parser.add_argument(
        "--player-log-interval",
        type=int,
        default=1,
        help=(
            "Print player logs every N poker rounds."
        ),
    )

    parser.add_argument(
        "--engine-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable internal PyPokerEngine logs."
        ),
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help=(
            "Print progress every N training games."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Override the number of training episodes "
            "from TrainingConfig."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed. Defaults to "
            "TrainingConfig.default_seed."
        ),
    )

    parser.add_argument(
        "--epsilon-schedule",
        choices=SUPPORTED_EPSILON_SCHEDULES,
        default=None,
        help=(
            "Override the epsilon schedule "
            "from TrainingConfig."
        ),
    )

    parser.add_argument(
        "--alpha-mode",
        choices=SUPPORTED_ALPHA_MODES,
        default=None,
        help=(
            "Override the Monte Carlo alpha mode "
            "from TrainingConfig. constant uses fixed alpha, "
            "visit_count uses 1/N(s,a), and sqrt_visit "
            "uses 1/sqrt(N(s,a))."
        ),
    )

    parser.add_argument(
        "--checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable model checkpoints."
        ),
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help=(
            "Save a checkpoint every N episodes. "
            "When omitted, checkpoint episodes "
            "from TrainingConfig are used."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Path for the final trained model. "
            "When omitted, the default TrainingConfig "
            "path is used."
        ),
    )

    parser.add_argument(
        "--checkpoint-directory",
        type=str,
        default=None,
        help=(
            "Directory used to store checkpoints."
        ),
    )

    parser.add_argument(
        "--checkpoint-episodes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit checkpoint episode numbers, "
            "for example: "
            "--checkpoint-episodes 1000 2500 5000."
        ),
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
            "--player-log-interval must be greater "
            "than zero"
        )

    if (
        args.episodes is not None
        and args.episodes <= 0
    ):
        parser.error(
            "--episodes must be greater than zero"
        )

    if (
        args.seed is not None
        and args.seed < 0
    ):
        parser.error(
            "--seed must be non-negative"
        )

    if (
        args.checkpoint_interval is not None
        and args.checkpoint_interval <= 0
    ):
        parser.error(
            "--checkpoint-interval must be greater "
            "than zero"
        )

    if args.checkpoint_episodes is not None:
        if any(
                episode <= 0
                for episode in args.checkpoint_episodes
        ):
            parser.error(
                "All --checkpoint-episodes values "
                "must be greater than zero"
            )

        if len(set(args.checkpoint_episodes)) != len(
                args.checkpoint_episodes
        ):
            parser.error(
                "--checkpoint-episodes must not "
                "contain duplicates"
            )

    return args


def parse_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a general Monte Carlo poker agent."
        ),
    )

    add_common_training_arguments(
        parser
    )

    args = parser.parse_args()

    return validate_training_args(
        parser,
        args,
    )


def parse_specialist_training_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Monte Carlo specialist against "
            "one fixed opponent type."
        ),
    )

    parser.add_argument(
        "--opponent",
        required=True,
        choices=TRAINING_OPPONENT_TYPES,
        help=(
            "Opponent type used during specialist "
            "training."
        ),
    )

    add_common_training_arguments(
        parser
    )

    args = parser.parse_args()

    return validate_training_args(
        parser,
        args,
    )
