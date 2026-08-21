from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.constants import CHECKPOINT_PREFIXES, MODEL_DIRECTORIES
from src.evaluation.runners.model_evaluator import (
    ModelBundle,
    TrainingOpponentEvaluationConfig,
    build_model_bundle_from_paths,
    discover_seed_directories,
    evaluate_training_opponent_bundle,
    parse_seed_from_directory,
    validate_model_paths,
    write_rows,
)


@dataclass(frozen=True)
class LearningCurveEvaluationConfig:
    games_per_matchup: int
    opponents: tuple[str, ...]
    tested_agents: tuple[str, ...]
    eval_seed_base: int
    output_path: Path


def checkpoint_filename(
    policy_type: str,
    checkpoint_episode: int,
    seed: int,
) -> str:
    prefix = CHECKPOINT_PREFIXES[policy_type]
    return f"{prefix}_episodes_{checkpoint_episode}_seed_{seed}.pkl"


def checkpoint_model_path(
    seed_directory: Path,
    policy_type: str,
    checkpoint_episode: int,
    seed: int,
) -> Path:
    return (
        seed_directory
        / MODEL_DIRECTORIES[policy_type]
        / "checkpoints"
        / checkpoint_filename(
            policy_type=policy_type,
            checkpoint_episode=checkpoint_episode,
            seed=seed,
        )
    )


def build_checkpoint_policy_paths(
    *,
    seed_directory: Path,
    checkpoint_episode: int,
    seed: int,
) -> dict[str, Path]:
    return {
        policy_type: checkpoint_model_path(
            seed_directory=seed_directory,
            policy_type=policy_type,
            checkpoint_episode=checkpoint_episode,
            seed=seed,
        )
        for policy_type in MODEL_DIRECTORIES
    }


def _optional_checkpoint_paths(
    *,
    run_directory: str | Path | None,
    seed: int,
    checkpoint_episode: int,
    bundle_name: str,
) -> dict[str, Path] | None:
    if run_directory is None:
        return None

    paths = build_checkpoint_policy_paths(
        seed_directory=Path(run_directory) / f"seed_{seed}",
        checkpoint_episode=checkpoint_episode,
        seed=seed,
    )
    validate_model_paths(paths.values(), bundle_name=bundle_name)
    return paths


def build_checkpoint_model_bundle(
    training_run_directory: str | Path,
    seed: int,
    checkpoint_episode: int,
    q_learning_run_directory: str | Path | None = None,
    sarsa_run_directory: str | Path | None = None,
    double_q_learning_run_directory: str | Path | None = None,
) -> ModelBundle:
    root = Path(training_run_directory)
    paths = build_checkpoint_policy_paths(
        seed_directory=root / f"seed_{seed}",
        checkpoint_episode=checkpoint_episode,
        seed=seed,
    )
    validate_model_paths(paths.values(), bundle_name="Monte Carlo")

    q_learning_paths = _optional_checkpoint_paths(
        run_directory=q_learning_run_directory,
        seed=seed,
        checkpoint_episode=checkpoint_episode,
        bundle_name="Q-learning",
    )
    sarsa_paths = _optional_checkpoint_paths(
        run_directory=sarsa_run_directory,
        seed=seed,
        checkpoint_episode=checkpoint_episode,
        bundle_name="SARSA",
    )
    double_q_learning_paths = _optional_checkpoint_paths(
        run_directory=double_q_learning_run_directory,
        seed=seed,
        checkpoint_episode=checkpoint_episode,
        bundle_name="Double Q-learning",
    )

    return build_model_bundle_from_paths(
        training_run_directory=root,
        seed=seed,
        episode=checkpoint_episode,
        model_source="checkpoint",
        paths=paths,
        q_learning_run_directory=q_learning_run_directory,
        q_learning_paths=q_learning_paths,
        sarsa_run_directory=sarsa_run_directory,
        sarsa_paths=sarsa_paths,
        double_q_learning_run_directory=double_q_learning_run_directory,
        double_q_learning_paths=double_q_learning_paths,
    )


def discover_checkpoint_model_bundles(
    training_run_directory: str | Path,
    checkpoint_episodes: Iterable[int],
    seeds: Iterable[int] | None = None,
    skip_incomplete: bool = True,
    q_learning_run_directory: str | Path | None = None,
    sarsa_run_directory: str | Path | None = None,
    double_q_learning_run_directory: str | Path | None = None,
) -> list[ModelBundle]:
    root = Path(training_run_directory)
    discovered_seeds = (
        [parse_seed_from_directory(path) for path in discover_seed_directories(root)]
        if seeds is None
        else list(seeds)
    )

    bundles: list[ModelBundle] = []
    for seed in discovered_seeds:
        for checkpoint_episode in checkpoint_episodes:
            try:
                bundle = build_checkpoint_model_bundle(
                    training_run_directory=root,
                    seed=seed,
                    checkpoint_episode=checkpoint_episode,
                    q_learning_run_directory=q_learning_run_directory,
                    sarsa_run_directory=sarsa_run_directory,
                    double_q_learning_run_directory=(double_q_learning_run_directory),
                )
            except FileNotFoundError:
                if skip_incomplete:
                    continue
                raise
            bundles.append(bundle)

    return bundles


def evaluate_learning_curve_bundle(
    *,
    bundle: ModelBundle,
    config: LearningCurveEvaluationConfig,
) -> list[dict]:
    return evaluate_training_opponent_bundle(
        bundle=bundle,
        config=TrainingOpponentEvaluationConfig(
            games_per_matchup=config.games_per_matchup,
            opponents=config.opponents,
            tested_agents=config.tested_agents,
            eval_seed_base=config.eval_seed_base,
            output_path=config.output_path,
        ),
    )


def write_learning_curve_rows(
    output_path: str | Path,
    rows: Iterable[dict],
) -> None:
    write_rows(output_path=output_path, rows=rows)
