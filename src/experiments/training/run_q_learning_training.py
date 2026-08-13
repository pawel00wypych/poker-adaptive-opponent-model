from pathlib import Path
from typing import Sequence

from src.experiments.training.td_cli import (
    TDTrainingCliSpec,
    model_directory as _model_directory,
    parse_td_training_args,
    run_td_cli,
    run_td_training,
)
from src.training.q_learning_trainer import (
    Q_LEARNING_ALGORITHM_NAME,
    Q_LEARNING_DISPLAY_NAME,
    model_run_name,
    run_q_learning_model_training,
)

Q_LEARNING_CLI_SPEC = TDTrainingCliSpec(
    algorithm_name=Q_LEARNING_ALGORITHM_NAME,
    display_name=Q_LEARNING_DISPLAY_NAME,
    default_output_dir="results/training_runs/q_learning",
    trainer_function=run_q_learning_model_training,
    model_run_name_function=model_run_name,
)


def parse_args():
    return parse_td_training_args(Q_LEARNING_CLI_SPEC)


def model_directory(
    *,
    output_dir: Path,
    seed: int,
    model_type: str,
) -> Path:
    return _model_directory(
        output_dir=output_dir,
        seed=seed,
        model_type=model_type,
        model_run_name_function=model_run_name,
    )


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
    return run_td_training(
        spec=Q_LEARNING_CLI_SPEC,
        episodes=episodes,
        seeds=seeds,
        models=models,
        epsilon_schedule=epsilon_schedule,
        alpha=alpha,
        gamma=gamma,
        output_dir=output_dir,
        checkpoint_episodes=checkpoint_episodes,
        checkpoints_enabled=checkpoints_enabled,
        checkpoint_interval=checkpoint_interval,
        progress=progress,
        player_verbose=player_verbose,
        player_log_interval=player_log_interval,
        engine_verbose=engine_verbose,
        log_interval=log_interval,
    )


if __name__ == "__main__":
    run_td_cli(Q_LEARNING_CLI_SPEC)
