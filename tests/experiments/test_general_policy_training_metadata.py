"""Metadata guarantees for Monte Carlo training.

Monte Carlo deliberately keeps its own episode loop rather than reusing
``td_trainer`` (see the module docstrings of the two training scripts), so
nothing structural stops the metadata schemas drifting apart. These tests are
what stops it.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.q_learning_agent import QLearningAgent
from src.config import GameConfig
from src.experiments.training.run_general_policy_training import (
    build_metadata,
    run_general_policy_training,
)
from src.experiments.training.run_specialist_training import build_training_metadata
from src.rl.model_io import load_model_metadata
from src.training.td_trainer import TDTrainingSpec, build_td_metadata
from src.training.training_metadata import save_json

COMMON = dict(
    completed_episodes=2,
    total_episodes=4,
    seed=1,
    epsilon_schedule="linear",
    alpha_mode="constant",
    game_config=GameConfig(),
    duration_seconds=1.5,
    total_hands=10,
)


def _monte_carlo_general_metadata():
    return build_metadata(
        agent=MonteCarloAgent(),
        opponent_counter=Counter({"tight": 2}),
        **COMMON,
    )


def _monte_carlo_specialist_metadata():
    return build_training_metadata(
        agent=MonteCarloAgent(),
        opponent_type="tight",
        **COMMON,
    )


def _td_metadata():
    spec = TDTrainingSpec.__new__(TDTrainingSpec)
    object.__setattr__(spec, "algorithm_name", "q_learning")

    return build_td_metadata(
        spec=spec,
        model_type="general_policy",
        opponent_type="mixed",
        completed_episodes=2,
        total_episodes=4,
        seed=1,
        epsilon_schedule="linear",
        agent=QLearningAgent(),
        game_config=GameConfig(),
        duration_seconds=1.5,
        total_hands=10,
        opponent_counter=Counter({"tight": 2}),
    )


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("general_policy", _monte_carlo_general_metadata),
        ("specialist", _monte_carlo_specialist_metadata),
    ],
)
def test_monte_carlo_metadata_covers_the_td_schema(name, factory):
    """Monte Carlo model files must record everything the TD ones do.

    Otherwise the artefacts cannot evidence that the compared algorithms were
    trained under equal conditions, which is what the algorithm comparison
    rests on.
    """
    missing = set(_td_metadata()) - set(factory())

    assert not missing, f"{name} metadata is missing TD keys: {sorted(missing)}"


@pytest.mark.parametrize(
    "factory",
    [_monte_carlo_general_metadata, _monte_carlo_specialist_metadata],
)
def test_monte_carlo_records_the_discount_factor(factory):
    """gamma exists on MonteCarloAgent and equals the TD agents' value.

    It was simply never written down, so the model files could not show that
    the algorithms shared a discount factor.
    """
    metadata = factory()

    assert metadata["gamma"] == MonteCarloAgent().gamma
    assert metadata["gamma"] == QLearningAgent().gamma


def test_general_policy_metadata_reports_throughput():
    metadata = _monte_carlo_general_metadata()

    assert metadata["mean_hands_per_episode"] == 10 / 2
    assert metadata["hands_per_second"] == 10 / 1.5


def test_throughput_is_zero_rather_than_dividing_by_zero():
    metadata = build_metadata(
        agent=MonteCarloAgent(),
        opponent_counter=Counter(),
        completed_episodes=0,
        total_episodes=4,
        seed=1,
        epsilon_schedule="linear",
        alpha_mode="constant",
        game_config=GameConfig(),
        duration_seconds=0.0,
        total_hands=0,
    )

    assert metadata["mean_hands_per_episode"] == 0.0
    assert metadata["hands_per_second"] == 0.0


def _run_short_training(tmp_path):
    return run_general_policy_training(
        episodes=2,
        seed=7,
        output_path=str(tmp_path / "final.pkl"),
        checkpoint_directory=str(tmp_path / "checkpoints"),
        checkpoint_episodes=[1],
        checkpoints_enabled=True,
        progress=False,
        player_verbose=False,
        engine_verbose=False,
    )


def _checkpoint_files(tmp_path):
    return sorted((tmp_path / "checkpoints").glob("*.pkl"))


def _episode_from_filename(checkpoint):
    """general_policy_episodes_1_seed_7.pkl -> 1"""
    parts = checkpoint.stem.split("_")
    return int(parts[parts.index("episodes") + 1])


def test_checkpoint_writes_a_json_sidecar(tmp_path, monkeypatch):
    """The specialist trainer and the TD trainer both write one; this did not.

    Nothing reads checkpoint sidecars today, so this breaks no current flow -
    it leaves the artefact tree non-uniform and Monte Carlo general-policy
    checkpoints unauditable without unpickling them.
    """
    monkeypatch.setattr(GameConfig, "max_round", 3, raising=False)

    _run_short_training(tmp_path)

    checkpoints = _checkpoint_files(tmp_path)
    assert checkpoints, "no checkpoint was written"

    for checkpoint in checkpoints:
        assert checkpoint.with_suffix(".json").exists(), checkpoint.name


def test_checkpoint_sidecar_matches_the_metadata_inside_the_pickle(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(GameConfig, "max_round", 3, raising=False)

    _run_short_training(tmp_path)

    for checkpoint in _checkpoint_files(tmp_path):
        sidecar = json.loads(
            checkpoint.with_suffix(".json").read_text(encoding="utf-8")
        )
        embedded = load_model_metadata(str(checkpoint), "Monte Carlo")

        assert sidecar == embedded


def test_every_checkpoint_sidecar_describes_its_own_checkpoint(
    tmp_path, monkeypatch
):
    """A sidecar that cannot identify its own checkpoint is not provenance.

    build_checkpoint_episodes always appends the final episode, so a run with
    ``checkpoint_episodes=[1]`` over 2 episodes produces two checkpoints. Each
    sidecar must describe the one it sits next to, not merely exist.
    """
    monkeypatch.setattr(GameConfig, "max_round", 3, raising=False)

    _run_short_training(tmp_path)

    checkpoints = _checkpoint_files(tmp_path)
    assert len(checkpoints) >= 2, "expected a mid-run and a final-episode checkpoint"

    for checkpoint in checkpoints:
        sidecar = json.loads(
            checkpoint.with_suffix(".json").read_text(encoding="utf-8")
        )

        assert sidecar["completed_episodes"] == _episode_from_filename(checkpoint)
        assert sidecar["seed"] == 7
        assert sidecar["algorithm"] == MonteCarloAgent.ALGORITHM_ID


def test_final_model_still_writes_its_sidecar(tmp_path, monkeypatch):
    """Guards against the checkpoint change disturbing the final-model path,
    which build_final_model_bundle does read."""
    monkeypatch.setattr(GameConfig, "max_round", 3, raising=False)

    _run_short_training(tmp_path)

    assert (tmp_path / "final.json").exists()


def test_save_json_round_trips(tmp_path):
    payload = _monte_carlo_general_metadata()
    path = Path(tmp_path) / "meta.json"

    save_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
