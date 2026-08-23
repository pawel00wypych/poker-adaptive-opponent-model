"""Where a training run writes its artefacts.

Every evaluator resolves a final model as
``<seed directory>/<policy directory>/final.pkl`` - see
``build_final_policy_paths`` in ``src/evaluation/runners/model_evaluator.py``.
A training default that omits the seed directory therefore produces artefacts
no evaluator can pick up, even though the run itself succeeds.

The orchestrators (``run_monte_carlo_suite.py`` and ``td_cli.py``) already build
the seeded layout themselves and pass an explicit output path. These helpers are
the fallback used when a training script is invoked directly, which is what the
README documents, so they have to produce the same layout.

The policy directory name is defined here for the training side and in
``MODEL_DIRECTORIES`` for the evaluation side. Training deliberately does not
import from evaluation, so the two are kept in step by
``tests/training/test_model_paths.py`` rather than by a shared import.
"""

from pathlib import Path

from src.poker.constants import TRAINING_OPPONENT_TYPES
from src.training.constants import MODEL_TYPE_GENERAL_POLICY

FINAL_MODEL_FILENAME = "final.pkl"
CHECKPOINT_DIRECTORY_NAME = "checkpoints"


def policy_directory_name(
    model_type: str,
    *,
    error_label: str = "training",
) -> str:
    """Directory a policy's artefacts live in, e.g. ``specialist_tight``."""
    if model_type == MODEL_TYPE_GENERAL_POLICY:
        return MODEL_TYPE_GENERAL_POLICY

    if model_type in TRAINING_OPPONENT_TYPES:
        return f"specialist_{model_type}"

    raise ValueError(
        f"Unsupported {error_label} model type: {model_type}"
    )


def seed_directory_name(seed: int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    return f"seed_{seed}"


def run_directory(
    *,
    model_root_directory: str | Path,
    seed: int,
    model_type: str,
    algorithm_key: str,
    error_label: str = "training",
) -> Path:
    """Directory holding one algorithm's artefacts for one seed and policy.

    ``<root>/<algorithm>/seed_<n>/<policy>/`` - the algorithm segment keeps the
    four algorithms from overwriting each other, and the seed segment keeps two
    runs of the same algorithm apart.
    """
    if not algorithm_key:
        raise ValueError("algorithm_key must not be empty")

    return (
        Path(model_root_directory)
        / algorithm_key
        / seed_directory_name(seed)
        / policy_directory_name(model_type, error_label=error_label)
    )


def default_final_model_path(
    *,
    model_root_directory: str | Path,
    seed: int,
    model_type: str,
    algorithm_key: str,
    error_label: str = "training",
) -> str:
    return str(
        run_directory(
            model_root_directory=model_root_directory,
            seed=seed,
            model_type=model_type,
            algorithm_key=algorithm_key,
            error_label=error_label,
        )
        / FINAL_MODEL_FILENAME
    )


def default_checkpoint_directory_path(
    *,
    model_root_directory: str | Path,
    seed: int,
    model_type: str,
    algorithm_key: str,
    error_label: str = "training",
) -> str:
    return str(
        run_directory(
            model_root_directory=model_root_directory,
            seed=seed,
            model_type=model_type,
            algorithm_key=algorithm_key,
            error_label=error_label,
        )
        / CHECKPOINT_DIRECTORY_NAME
    )
