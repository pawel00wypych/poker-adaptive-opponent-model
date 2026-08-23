"""Tests for process-wide seeding.

This module had no test file before, which is part of why the no-op
``PYTHONHASHSEED`` assignment survived for so long.
"""

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from src.training.random_utils import set_global_seed

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

HASH_PROBE = (
    "import sys; sys.path.insert(0, r'{root}'); "
    "from src.training.random_utils import set_global_seed; "
    "set_global_seed(123); print(hash('tight'))"
)


def _run_probe(hash_seed: str | None) -> str:
    environment = dict(os.environ)

    if hash_seed is None:
        environment.pop("PYTHONHASHSEED", None)
    else:
        environment["PYTHONHASHSEED"] = hash_seed

    completed = subprocess.run(
        [sys.executable, "-c", HASH_PROBE.format(root=REPOSITORY_ROOT)],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    return completed.stdout.strip()


def test_set_global_seed_does_not_write_hash_seed_into_the_environment(monkeypatch):
    """CPython reads PYTHONHASHSEED only at startup, so setting it here is a lie."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    set_global_seed(7)

    assert "PYTHONHASHSEED" not in os.environ


def test_set_global_seed_makes_the_global_random_module_reproducible():
    set_global_seed(11)
    first = [random.random() for _ in range(5)]

    set_global_seed(11)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_different_seeds_produce_different_sequences():
    set_global_seed(11)
    first = [random.random() for _ in range(5)]

    set_global_seed(12)
    second = [random.random() for _ in range(5)]

    assert first != second


@pytest.mark.parametrize("seed", [-1, -42])
def test_set_global_seed_rejects_a_negative_seed(seed):
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seed(seed)


def test_hash_seed_only_takes_effect_when_set_before_interpreter_start():
    """Documents the CPython semantics that make the deleted assignment a no-op.

    Two child interpreters that both call ``set_global_seed(123)`` hash the same
    string differently unless PYTHONHASHSEED was already in the environment when
    they started. This is why a launcher has to set it for a child process, as
    ``run_monte_carlo_suite.py`` does.
    """
    pinned = {_run_probe("0") for _ in range(2)}
    assert len(pinned) == 1, "a startup-set PYTHONHASHSEED must pin string hashing"

    unpinned = {_run_probe(None) for _ in range(4)}
    assert len(unpinned) > 1, (
        "without a startup-set PYTHONHASHSEED, calling set_global_seed "
        "cannot pin string hashing"
    )
