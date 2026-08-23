from src.agents.sarsa_agent import SarsaAgent
from src.training.td_trainer import (
    TDTrainingSpec,
    build_td_metadata,
    format_duration,
    run_td_model_training,
)
from src.training.td_trainer import (
    build_episode_opponent as _build_episode_opponent,
)
from src.training.td_trainer import (
    build_training_player as _build_training_player,
)
from src.training.td_trainer import (
    checkpoint_model_name as _checkpoint_model_name,
)
from src.training.td_trainer import (
    default_checkpoint_directory as _default_checkpoint_directory,
)
from src.training.td_trainer import (
    default_model_path as _default_model_path,
)
from src.training.td_trainer import (
    model_run_name as _model_run_name,
)

SARSA_ALGORITHM_NAME = "sarsa"
SARSA_DISPLAY_NAME = "SARSA"

SARSA_TRAINING_SPEC = TDTrainingSpec(
    algorithm_name=SARSA_ALGORITHM_NAME,
    display_name=SARSA_DISPLAY_NAME,
    agent_factory=SarsaAgent,
    player_name_suffix="sarsa",
    registered_player_name="sarsa",
)


def model_run_name(model_type: str) -> str:
    return _model_run_name(
        model_type,
        error_label=SARSA_DISPLAY_NAME,
    )


def default_model_path(model_type: str, seed: int) -> str:
    return _default_model_path(
        spec=SARSA_TRAINING_SPEC,
        model_type=model_type,
        seed=seed,
    )


def default_checkpoint_directory(model_type: str, seed: int) -> str:
    return _default_checkpoint_directory(
        spec=SARSA_TRAINING_SPEC,
        model_type=model_type,
        seed=seed,
    )


def checkpoint_model_name(model_type: str) -> str:
    return _checkpoint_model_name(
        model_type,
        error_label=SARSA_DISPLAY_NAME,
    )


def build_sarsa_metadata(**kwargs) -> dict:
    return build_td_metadata(
        spec=SARSA_TRAINING_SPEC,
        **kwargs,
    )


def build_training_player(**kwargs):
    return _build_training_player(
        spec=SARSA_TRAINING_SPEC,
        **kwargs,
    )


def build_episode_opponent(**kwargs):
    return _build_episode_opponent(
        error_label=SARSA_DISPLAY_NAME,
        **kwargs,
    )


def run_sarsa_model_training(**kwargs) -> dict:
    return run_td_model_training(
        spec=SARSA_TRAINING_SPEC,
        **kwargs,
    )


__all__ = [
    "SARSA_ALGORITHM_NAME",
    "SARSA_TRAINING_SPEC",
    "build_episode_opponent",
    "build_sarsa_metadata",
    "build_training_player",
    "checkpoint_model_name",
    "default_checkpoint_directory",
    "default_model_path",
    "format_duration",
    "model_run_name",
    "run_sarsa_model_training",
]
