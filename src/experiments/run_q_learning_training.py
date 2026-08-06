import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

from src.experiments.constants import (
    MODEL_TYPE_SINGLE_POLICY,
    MODEL_TYPES,
)
from src.training.constants import SUPPORTED_EPSILON_SCHEDULES
from src.training.q_learning_trainer import (
    format_duration,
    model_run_name,
    run_q_learning_model_training,
)
from src.training.training_metadata import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train tabular Q-learning poker policies. "
            "By default this trains the general single policy "
            "and all specialist policies."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=2_000,
        help="Number of training episodes per model.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds used for independent Q-learning runs.",
    )

    parser.add_argument(
        "--models",
        choices=MODEL_TYPES,
        nargs="+",
        default=list(MODEL_TYPES),
        help=(
            "Models to train: single_policy or one of the "
            "specialist opponent types."
        ),
    )

    parser.add_argument(
        "--epsilon-schedule",
        choices=SUPPORTED_EPSILON_SCHEDULES,
        default=SUPPORTED_EPSILON_SCHEDULES[0],
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override TrainingConfig.alpha for Q-learning.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Q-learning discount factor.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/training_runs/q_learning",
        help=(
            "Root directory where Q-learning models are saved. "
            "The structure matches the existing checkpoint layout."
        ),
    )

    parser.add_argument(
        "--checkpoint-episodes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit checkpoint episode numbers. When omitted, "
            "TrainingConfig.checkpoint_episodes are used."
        ),
    )

    parser.add_argument(
        "--checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Q-learning model checkpoints.",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save a checkpoint every N episodes.",
    )

    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic progress logs.",
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
        help="Print player logs every N poker rounds.",
    )

    parser.add_argument(
        "--engine-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable internal PyPokerEngine logs.",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print progress every N training games.",
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    if args.gamma < 0 or args.gamma > 1:
        parser.error("--gamma must be in range [0, 1]")

    if args.alpha is not None and not 0 < args.alpha <= 1:
        parser.error("--alpha must be in range (0, 1]")

    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be non-negative")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    if args.log_interval <= 0:
        parser.error("--log-interval must be greater than zero")

    if args.player_log_interval <= 0:
        parser.error(
            "--player-log-interval must be greater than zero"
        )

    if (
        args.checkpoint_interval is not None
        and args.checkpoint_interval <= 0
    ):
        parser.error(
            "--checkpoint-interval must be greater than zero"
        )

    if args.checkpoint_episodes is not None:
        if any(
            episode <= 0
            for episode in args.checkpoint_episodes
        ):
            parser.error(
                "All --checkpoint-episodes values must be greater than zero"
            )

        if len(set(args.checkpoint_episodes)) != len(
            args.checkpoint_episodes
        ):
            parser.error(
                "--checkpoint-episodes must not contain duplicates"
            )

    return args


def model_directory(
    *,
    output_dir: Path,
    seed: int,
    model_type: str,
) -> Path:
    return output_dir / f"seed_{seed}" / model_run_name(model_type)


def run_q_learning_training(
    *,
    episodes: int,
    seeds: Sequence[int],
    models: Sequence[str],
    epsilon_schedule: str,
    alpha: float | None,
    gamma: float,
    output_dir: str,
    checkpoint_episodes: Sequence[int] | None,
    checkpoints_enabled: bool,
    checkpoint_interval: int | None,
    progress: bool,
    player_verbose: bool,
    player_log_interval: int,
    engine_verbose: bool,
    log_interval: int,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    started_at = perf_counter()
    results: list[dict] = []

    for seed in seeds:
        for model_type in models:
            run_directory = model_directory(
                output_dir=output_root,
                seed=seed,
                model_type=model_type,
            )
            final_model_path = run_directory / "final.pkl"
            checkpoint_directory = run_directory / "checkpoints"

            metadata = run_q_learning_model_training(
                model_type=model_type,
                episodes=episodes,
                seed=seed,
                epsilon_schedule=epsilon_schedule,
                alpha=alpha,
                gamma=gamma,
                output_path=str(final_model_path),
                checkpoint_directory=str(checkpoint_directory),
                checkpoint_episodes=checkpoint_episodes,
                checkpoints_enabled=checkpoints_enabled,
                checkpoint_interval=checkpoint_interval,
                progress=progress,
                player_verbose=player_verbose,
                player_log_interval=player_log_interval,
                engine_verbose=engine_verbose,
                log_interval=log_interval,
            )

            results.append(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "final_model_path": str(final_model_path),
                    "checkpoint_directory": str(checkpoint_directory),
                    "metadata": metadata,
                }
            )

    duration_seconds = perf_counter() - started_at

    summary = {
        "algorithm": "q_learning",
        "episodes": episodes,
        "seeds": list(seeds),
        "models": list(models),
        "epsilon_schedule": epsilon_schedule,
        "alpha": alpha,
        "gamma": gamma,
        "output_dir": str(output_root),
        "duration_seconds": duration_seconds,
        "duration": format_duration(duration_seconds),
        "jobs": results,
    }

    save_json(
        output_root / "training_summary.json",
        summary,
    )

    print(
        "Q-learning training suite finished\n"
        f"episodes={episodes}\n"
        f"seeds={list(seeds)}\n"
        f"models={list(models)}\n"
        f"duration={format_duration(duration_seconds)}\n"
        f"output_dir={output_root}"
    )

    return summary


if __name__ == "__main__":
    cli_args = parse_args()

    run_q_learning_training(
        episodes=cli_args.episodes,
        seeds=cli_args.seeds,
        models=cli_args.models,
        epsilon_schedule=cli_args.epsilon_schedule,
        alpha=cli_args.alpha,
        gamma=cli_args.gamma,
        output_dir=cli_args.output_dir,
        checkpoint_episodes=cli_args.checkpoint_episodes,
        checkpoints_enabled=cli_args.checkpoints,
        checkpoint_interval=cli_args.checkpoint_interval,
        progress=cli_args.progress,
        player_verbose=cli_args.player_verbose,
        player_log_interval=cli_args.player_log_interval,
        engine_verbose=cli_args.engine_verbose,
        log_interval=cli_args.log_interval,
    )
