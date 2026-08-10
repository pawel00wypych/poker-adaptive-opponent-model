from pathlib import Path

import pytest

from src.agents.q_learning_agent import QLearningAgent
from src.experiments.constants import MODEL_TYPE_GENERAL_POLICY
from src.poker.constants import OPPONENT_TYPE_CALLING
from src.training.q_learning_trainer import (
    build_q_learning_metadata,
    default_checkpoint_directory,
    default_model_path,
    model_run_name,
    run_q_learning_model_training,
)


def test_model_run_name_supports_general_policy_and_specialists():
    assert model_run_name(MODEL_TYPE_GENERAL_POLICY) == "general_policy"
    assert model_run_name(OPPONENT_TYPE_CALLING) == "specialist_calling"


def test_model_run_name_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Unsupported Q-learning model type"):
        model_run_name("balanced")


def test_build_q_learning_metadata_contains_algorithm_fields():
    class GameConfigStub:
        max_round = 100
        initial_stack = 200
        small_blind_amount = 5

    agent = QLearningAgent(
        alpha=0.2,
        gamma=0.95,
        epsilon=0.1,
        epsilon_min=0.05,
    )

    metadata = build_q_learning_metadata(
        model_type=MODEL_TYPE_GENERAL_POLICY,
        opponent_type="mixed",
        completed_episodes=2,
        total_episodes=10,
        seed=42,
        epsilon_schedule="linear",
        agent=agent,
        game_config=GameConfigStub(),
        duration_seconds=4.0,
        total_hands=8,
    )

    assert metadata["algorithm"] == "q_learning"
    assert metadata["model_type"] == MODEL_TYPE_GENERAL_POLICY
    assert metadata["opponent_type"] == "mixed"
    assert metadata["alpha"] == pytest.approx(0.2)
    assert metadata["gamma"] == pytest.approx(0.95)
    assert metadata["mean_hands_per_episode"] == pytest.approx(4.0)
    assert metadata["hands_per_second"] == pytest.approx(2.0)


def test_run_q_learning_model_training_smoke(tmp_path):
    output_path = tmp_path / "final.pkl"
    checkpoint_dir = tmp_path / "checkpoints"

    metadata = run_q_learning_model_training(
        model_type=MODEL_TYPE_GENERAL_POLICY,
        episodes=1,
        seed=42,
        output_path=str(output_path),
        checkpoint_directory=str(checkpoint_dir),
        checkpoint_episodes=[1],
        progress=False,
        player_verbose=False,
        engine_verbose=False,
        log_interval=1,
    )

    checkpoint_path = checkpoint_dir / "general_policy_episodes_1_seed_42.pkl"

    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".json").exists()
    assert metadata["algorithm"] == "q_learning"
    assert metadata["completed_episodes"] == 1
    assert metadata["seed"] == 42
    assert metadata["states"] >= 0
