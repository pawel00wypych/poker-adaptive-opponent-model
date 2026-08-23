"""The training budget must be identical across all four algorithms.

Episodes and seeds are the two things that decide how much training each
algorithm gets. If they differ, the algorithm comparison measures the budget
rather than the algorithm - which is the confound PR-9 was supposed to remove.

The seed count is load-bearing in a second way: seed-level statistics are the
only basis the thesis uses for a claim, and they are undefined below two seeds.
"""

import sys

import pandas as pd
import pytest

from src.config import TrainingConfig
from src.evaluation.metrics.seed_statistics import (
    SEED_CI_LOWER_COLUMN,
    SEED_CI_UPPER_COLUMN,
    add_seed_level_statistical_summary,
)
from src.experiments.training.run_double_q_learning_training import (
    parse_args as parse_double_q_learning_args,
)
from src.experiments.training.run_monte_carlo_suite import (
    parse_args as parse_monte_carlo_args,
)
from src.experiments.training.run_q_learning_training import (
    parse_args as parse_q_learning_args,
)
from src.experiments.training.run_sarsa_training import parse_args as parse_sarsa_args

ORCHESTRATORS = {
    "monte_carlo": parse_monte_carlo_args,
    "q_learning": parse_q_learning_args,
    "sarsa": parse_sarsa_args,
    "double_q_learning": parse_double_q_learning_args,
}

# Named as literals so this module still imports where the presets do not yet
# exist, which keeps the seed-parity failure readable instead of a collection
# error.
PRESET_NAMES = ("final", "verification")


def _parse(parser, argv):
    sys.argv = ["training", *argv]
    return parser()


def _budget(namespace):
    return (
        namespace.episodes,
        tuple(namespace.seeds),
        tuple(sorted(namespace.checkpoint_episodes)),
    )


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_every_orchestrator_uses_the_same_budget(preset):
    """The bug: the Monte Carlo suite defaulted to 5 seeds and td_cli to 1.

    A default run therefore compared 5 seeds of Monte Carlo evidence against a
    single seed of each TD algorithm.
    """
    budgets = {
        name: _budget(_parse(parser, ["--config", preset]))
        for name, parser in ORCHESTRATORS.items()
    }

    assert len(set(budgets.values())) == 1, budgets


@pytest.mark.parametrize("name", sorted(ORCHESTRATORS))
def test_no_orchestrator_defaults_to_a_single_seed(name):
    """A single-seed run cannot produce a seed-level confidence interval."""
    namespace = _parse(ORCHESTRATORS[name], [])

    assert len(namespace.seeds) >= 2


def test_a_single_seed_cannot_support_a_claim():
    """Why the seed count is not a cosmetic default.

    Pins the reason: below two seeds the authoritative interval is NaN, so a
    single-seed algorithm silently drops out of every comparison.
    """
    widths = {}

    for seed_count in (1, 2, 5):
        aggregated = add_seed_level_statistical_summary(
            pd.DataFrame(
                [
                    {
                        "agent_name": "adaptive_mc",
                        "opponent_name": "tight",
                        "seeds": seed_count,
                        "mean_profit_bb": 5.0,
                        "mean_profit_bb_std_across_seeds": (
                            0.0 if seed_count == 1 else 1.2
                        ),
                    }
                ]
            )
        ).iloc[0]

        widths[seed_count] = (
            aggregated[SEED_CI_LOWER_COLUMN],
            aggregated[SEED_CI_UPPER_COLUMN],
        )

    assert pd.isna(widths[1][0]) and pd.isna(widths[1][1])
    assert pd.notna(widths[2][0]) and pd.notna(widths[2][1])
    assert pd.notna(widths[5][0]) and pd.notna(widths[5][1])


@pytest.mark.parametrize("name", sorted(ORCHESTRATORS))
def test_explicit_flags_override_the_preset(name):
    namespace = _parse(
        ORCHESTRATORS[name],
        [
            "--config",
            "verification",
            "--episodes",
            "700",
            "--seeds",
            "1",
            "2",
        ],
    )

    assert namespace.episodes == 700
    assert list(namespace.seeds) == [1, 2]


@pytest.mark.parametrize("name", sorted(ORCHESTRATORS))
def test_an_episode_budget_below_the_preset_checkpoints_is_rejected(name):
    """Silently dropping checkpoints would make two algorithms produce
    different diagnostics from the same flags."""
    with pytest.raises(SystemExit):
        _parse(
            ORCHESTRATORS[name],
            ["--config", "verification", "--episodes", "7"],
        )


@pytest.mark.parametrize("name", sorted(ORCHESTRATORS))
def test_the_default_preset_is_the_final_experiment(name):
    without_flag = _budget(_parse(ORCHESTRATORS[name], []))
    explicit = _budget(_parse(ORCHESTRATORS[name], ["--config", "final"]))

    assert without_flag == explicit


def test_the_verification_preset_is_a_small_fraction_of_the_final_one():
    """It has to be cheap enough that someone actually runs it first."""
    from src.config import FINAL_CONFIG, VERIFICATION_CONFIG

    final_cost = FINAL_CONFIG.episodes * len(FINAL_CONFIG.seeds)
    verification_cost = VERIFICATION_CONFIG.episodes * len(VERIFICATION_CONFIG.seeds)

    assert verification_cost < final_cost / 20


def test_the_verification_preset_still_produces_usable_statistics():
    """A rehearsal that cannot yield a confidence interval would not rehearse
    the reporting layer it exists to exercise."""
    from src.config import VERIFICATION_CONFIG

    assert len(VERIFICATION_CONFIG.seeds) >= 2


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_checkpoints_stay_inside_the_budget(preset):
    from src.config import training_config_for

    config = training_config_for(preset)

    assert max(config.checkpoint_episodes) <= config.episodes


def test_the_default_preset_name_resolves():
    from src.config import (
        DEFAULT_TRAINING_CONFIG_PRESET,
        FINAL_CONFIG,
        training_config_for,
    )

    assert training_config_for(DEFAULT_TRAINING_CONFIG_PRESET) is FINAL_CONFIG


def test_an_unknown_preset_is_rejected():
    from src.config import training_config_for

    with pytest.raises(ValueError, match="Unsupported training config preset"):
        training_config_for("nonsense")


def test_default_seed_is_derived_from_the_seed_set():
    """Two hand-maintained fields drift; a derived one cannot."""
    from src.config import VERIFICATION_CONFIG

    assert TrainingConfig().default_seed == TrainingConfig().seeds[0]
    assert VERIFICATION_CONFIG.default_seed == VERIFICATION_CONFIG.seeds[0]


def test_checkpoints_beyond_the_budget_are_rejected():
    with pytest.raises(ValueError, match="must not exceed the training budget"):
        TrainingConfig(episodes=100, checkpoint_episodes=(50, 200))


def test_duplicate_seeds_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        TrainingConfig(seeds=(42, 42))


def test_an_empty_seed_set_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        TrainingConfig(seeds=())


def test_a_non_positive_episode_budget_is_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        TrainingConfig(episodes=0)
