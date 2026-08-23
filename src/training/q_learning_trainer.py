from src.agents.q_learning_agent import QLearningAgent
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

Q_LEARNING_ALGORITHM_NAME = "q_learning"
Q_LEARNING_DISPLAY_NAME = "Q-learning"

Q_LEARNING_TRAINING_SPEC = TDTrainingSpec(
    algorithm_name=Q_LEARNING_ALGORITHM_NAME,
    display_name=Q_LEARNING_DISPLAY_NAME,
    agent_factory=QLearningAgent,
    player_name_suffix="q_learning",
    registered_player_name="q_learning",
)


def model_run_name(model_type: str) -> str:
    return _model_run_name(
        model_type,
        error_label=Q_LEARNING_DISPLAY_NAME,
    )


def default_model_path(model_type: str, seed: int) -> str:
    return _default_model_path(
        spec=Q_LEARNING_TRAINING_SPEC,
        model_type=model_type,
        seed=seed,
    )


def default_checkpoint_directory(model_type: str, seed: int) -> str:
    return _default_checkpoint_directory(
        spec=Q_LEARNING_TRAINING_SPEC,
        model_type=model_type,
        seed=seed,
    )


def checkpoint_model_name(model_type: str) -> str:
    return _checkpoint_model_name(
        model_type,
        error_label=Q_LEARNING_DISPLAY_NAME,
    )


def build_q_learning_metadata(**kwargs) -> dict:
    return build_td_metadata(
        spec=Q_LEARNING_TRAINING_SPEC,
        **kwargs,
    )


def build_training_player(**kwargs):
    return _build_training_player(
        spec=Q_LEARNING_TRAINING_SPEC,
        **kwargs,
    )


def build_episode_opponent(**kwargs):
    return _build_episode_opponent(
        error_label=Q_LEARNING_DISPLAY_NAME,
        **kwargs,
    )


def run_q_learning_model_training(**kwargs) -> dict:
    return run_td_model_training(
        spec=Q_LEARNING_TRAINING_SPEC,
        **kwargs,
    )


__all__ = [
    "Q_LEARNING_ALGORITHM_NAME",
    "Q_LEARNING_TRAINING_SPEC",
    "build_episode_opponent",
    "build_q_learning_metadata",
    "build_training_player",
    "checkpoint_model_name",
    "default_checkpoint_directory",
    "default_model_path",
    "format_duration",
    "model_run_name",
    "run_q_learning_model_training",
]
