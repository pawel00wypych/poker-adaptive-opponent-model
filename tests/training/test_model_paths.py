"""Training must write where evaluation reads.

The orchestrators (``run_monte_carlo_suite.py``, ``td_cli.py``) build the seeded
layout themselves. These tests cover the fallback used when a training script is
invoked directly, which is what the README documents - and which previously
produced artefacts no evaluator could resolve.
"""

from pathlib import Path

import pytest

from src.config import TrainingConfig
from src.evaluation.constants import MODEL_DIRECTORIES
from src.evaluation.runners.model_evaluator import build_final_policy_paths
from src.experiments.training.run_specialist_training import get_default_model_path
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
    OPPONENT_TYPE_UNKNOWN,
)
from src.training.constants import MODEL_TYPE_GENERAL_POLICY
from src.training.double_q_learning_trainer import (
    default_model_path as double_q_learning_model_path,
)
from src.training.model_paths import (
    default_checkpoint_directory_path,
    default_final_model_path,
    policy_directory_name,
    seed_directory_name,
)
from src.training.q_learning_trainer import default_model_path as q_learning_model_path
from src.training.sarsa_trainer import default_model_path as sarsa_model_path

SPECIALIST_TYPES = (
    OPPONENT_TYPE_TIGHT,
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
)
ALL_MODEL_TYPES = (MODEL_TYPE_GENERAL_POLICY, *SPECIALIST_TYPES)


def _monte_carlo_path(model_type, seed=42):
    config = TrainingConfig()

    if model_type == MODEL_TYPE_GENERAL_POLICY:
        return default_final_model_path(
            model_root_directory=config.model_root_directory,
            seed=seed,
            model_type=model_type,
        )

    return get_default_model_path(config, model_type, seed)


ALGORITHM_DEFAULT_PATHS = {
    "monte_carlo": _monte_carlo_path,
    "q_learning": q_learning_model_path,
    "sarsa": sarsa_model_path,
    "double_q_learning": double_q_learning_model_path,
}


@pytest.mark.parametrize("algorithm", sorted(ALGORITHM_DEFAULT_PATHS))
def test_every_algorithm_default_includes_the_seed_directory(algorithm):
    """Without the seed directory the evaluator cannot resolve the model.

    This was true of all four algorithms, not just Monte Carlo.
    """
    path = Path(ALGORITHM_DEFAULT_PATHS[algorithm](MODEL_TYPE_GENERAL_POLICY, 42))

    assert "seed_42" in path.parts


@pytest.mark.parametrize("algorithm", sorted(ALGORITHM_DEFAULT_PATHS))
def test_every_algorithm_default_is_a_final_model(algorithm):
    path = Path(ALGORITHM_DEFAULT_PATHS[algorithm](MODEL_TYPE_GENERAL_POLICY, 42))

    assert path.name == "final.pkl"


@pytest.mark.parametrize("model_type", ALL_MODEL_TYPES)
def test_no_default_output_path_points_at_a_checkpoint(model_type):
    """A final-model default must never resolve into checkpoints/.

    The calling specialist used to default to
    results/models/checkpoints/monte_carlo_vs_calling_episodes_7500_seed_42.pkl,
    contradicting the rule that final benchmarks use final.pkl only.
    """
    path = Path(_monte_carlo_path(model_type))

    assert "checkpoints" not in path.parts


def test_monte_carlo_defaults_land_exactly_where_the_evaluator_looks():
    """The end-to-end property: every policy the evaluator resolves is produced."""
    config = TrainingConfig()
    seed_directory = Path(config.model_root_directory) / "seed_42"
    expected = build_final_policy_paths(seed_directory=seed_directory)

    produced = {
        OPPONENT_TYPE_UNKNOWN: _monte_carlo_path(MODEL_TYPE_GENERAL_POLICY),
        OPPONENT_TYPE_TIGHT: _monte_carlo_path(OPPONENT_TYPE_TIGHT),
        OPPONENT_TYPE_AGGRESSIVE: _monte_carlo_path(OPPONENT_TYPE_AGGRESSIVE),
        OPPONENT_TYPE_CALLING: _monte_carlo_path(OPPONENT_TYPE_CALLING),
    }

    for policy_type, expected_path in expected.items():
        assert Path(produced[policy_type]) == expected_path, policy_type


def test_the_training_and_evaluation_directory_names_agree():
    """Two independent definitions of the same names, kept in step by this test.

    src/training deliberately does not import from src/evaluation, so nothing in
    the source enforces that policy_directory_name and MODEL_DIRECTORIES match.
    This is what does.
    """
    assert policy_directory_name(MODEL_TYPE_GENERAL_POLICY) == (
        MODEL_DIRECTORIES[OPPONENT_TYPE_UNKNOWN]
    )

    for opponent_type in SPECIALIST_TYPES:
        assert policy_directory_name(opponent_type) == MODEL_DIRECTORIES[opponent_type]


def test_checkpoint_defaults_are_also_seeded_and_per_policy():
    """A flat checkpoint directory is unreadable by the learning-curve evaluator,
    which resolves seed_<n>/<policy>/checkpoints/."""
    path = Path(
        default_checkpoint_directory_path(
            model_root_directory="results/models/monte_carlo",
            seed=7,
            model_type=OPPONENT_TYPE_TIGHT,
        )
    )

    assert path.parts[-3:] == ("seed_7", "specialist_tight", "checkpoints")


def test_different_seeds_do_not_share_a_directory():
    """The flat default made two seeds overwrite each other."""
    first = _monte_carlo_path(OPPONENT_TYPE_TIGHT, seed=1)
    second = _monte_carlo_path(OPPONENT_TYPE_TIGHT, seed=2)

    assert first != second


def test_different_policies_do_not_share_a_directory():
    paths = {_monte_carlo_path(model_type) for model_type in ALL_MODEL_TYPES}

    assert len(paths) == len(ALL_MODEL_TYPES)


def test_algorithms_do_not_share_a_root():
    roots = {
        Path(builder(MODEL_TYPE_GENERAL_POLICY, 42)).parts[2]
        for builder in ALGORITHM_DEFAULT_PATHS.values()
    }

    assert len(roots) == len(ALGORITHM_DEFAULT_PATHS)


@pytest.mark.parametrize("seed", [-1, -100])
def test_seed_directory_rejects_a_negative_seed(seed):
    with pytest.raises(ValueError):
        seed_directory_name(seed)


@pytest.mark.parametrize("seed", [1.5, "3", None, True])
def test_seed_directory_rejects_a_non_integer_seed(seed):
    with pytest.raises(TypeError):
        seed_directory_name(seed)


def test_policy_directory_name_rejects_an_unknown_model_type():
    with pytest.raises(ValueError, match="Unsupported training model type"):
        policy_directory_name("nonsense")


def test_the_removed_flat_path_fields_are_gone():
    """They looked like configuration but produced unusable artefacts."""
    config = TrainingConfig()

    for name in (
        "general_policy_model_path",
        "tight_model_path",
        "aggressive_model_path",
        "calling_model_path",
        "checkpoint_directory",
    ):
        assert not hasattr(config, name), name
