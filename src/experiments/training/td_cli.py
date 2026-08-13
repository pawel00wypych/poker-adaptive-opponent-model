import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from src.experiments.constants import MODEL_TYPES
from src.training.constants import SUPPORTED_EPSILON_SCHEDULES
from src.training.td_trainer import format_duration
from src.training.training_metadata import save_json


@dataclass(frozen=True)
class TDTrainingCliSpec:
    algorithm_name: str
    display_name: str
    default_output_dir: str
    trainer_function: Callable[..., dict]
    model_run_name_function: Callable[[str], str]


def parse_td_training_args(
    spec: TDTrainingCliSpec,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Train tabular {spec.display_name} poker policies. "
            "By default this trains the general policy and all "
            "specialist policies."
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
        help=(
            "Random seeds used for independent "
            f"{spec.display_name} runs."
        ),
    )

    parser.add_argument(
        "--models",
        choices=MODEL_TYPES,
        nargs="+",
        default=list(MODEL_TYPES),
        help=(
            "Models to train: general_policy or one of the "
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
        help=f"Override TrainingConfig.alpha for {spec.display_name}.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help=f"{spec.display_name} discount factor.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=spec.default_output_dir,
        help=(
            f"Root directory where {spec.display_name} models are saved. "
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
        help=f"Enable {spec.display_name} model checkpoints.",
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
    validate_td_training_args(
        parser=parser,
        args=args,
    )

    return args


def validate_td_training_args(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
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


def model_directory(
    *,
    output_dir: Path,
    seed: int,
    model_type: str,
    model_run_name_function: Callable[[str], str],
) -> Path:
    return (
        output_dir
        / f"seed_{seed}"
        / model_run_name_function(model_type)
    )


def run_td_training(
    *,
    spec: TDTrainingCliSpec,
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
                model_run_name_function=spec.model_run_name_function,
            )
            final_model_path = run_directory / "final.pkl"
            checkpoint_directory = run_directory / "checkpoints"

            metadata = spec.trainer_function(
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
        "algorithm": spec.algorithm_name,
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
        f"{spec.display_name} training suite finished\n"
        f"episodes={episodes}\n"
        f"seeds={list(seeds)}\n"
        f"models={list(models)}\n"
        f"duration={format_duration(duration_seconds)}\n"
        f"output_dir={output_root}"
    )

    return summary


def run_td_cli(spec: TDTrainingCliSpec) -> dict:
    args = parse_td_training_args(spec)

    return run_td_training(
        spec=spec,
        episodes=args.episodes,
        seeds=args.seeds,
        models=args.models,
        epsilon_schedule=args.epsilon_schedule,
        alpha=args.alpha,
        gamma=args.gamma,
        output_dir=args.output_dir,
        checkpoint_episodes=args.checkpoint_episodes,
        checkpoints_enabled=args.checkpoints,
        checkpoint_interval=args.checkpoint_interval,
        progress=args.progress,
        player_verbose=args.player_verbose,
        player_log_interval=args.player_log_interval,
        engine_verbose=args.engine_verbose,
        log_interval=args.log_interval,
    )
