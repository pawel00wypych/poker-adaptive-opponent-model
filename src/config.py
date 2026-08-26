from dataclasses import dataclass, replace

from src.training.constants import (
    ALPHA_MODE_SQRT_VISIT,
    EPSILON_SCHEDULE_LINEAR,
)


@dataclass(frozen=True)
class GameConfig:
    max_round: int = 100
    initial_stack: int = 200
    small_blind_amount: int = 5

    @property
    def big_blind_amount(self) -> int:
        return self.small_blind_amount * 2


@dataclass(frozen=True)
class TrainingConfig:
    """Training budget and artefact layout, shared by every algorithm.

    ``episodes`` and ``seeds`` are deliberately here rather than on each
    trainer. They are the two things that must be identical across Monte Carlo,
    Q-learning, SARSA and Double Q-learning for the algorithm comparison to
    measure the algorithm instead of how much training each one happened to get.

    ``seeds`` in particular is load-bearing. Seed-level statistics are the only
    basis the thesis uses for a claim, and they are undefined for fewer than two
    seeds - a single-seed run reports ``NaN`` for its confidence interval, so it
    cannot support any comparison at all.
    """

    episodes: int = 10_000
    alpha: float = 0.1
    alpha_mode: str = ALPHA_MODE_SQRT_VISIT

    epsilon_start: float = 0.5
    epsilon_min: float = 0.05
    epsilon_schedule: str = EPSILON_SCHEDULE_LINEAR

    seeds: tuple[int, ...] = (42, 123, 456, 789, 2026)

    checkpoint_episodes: tuple[int, ...] = (
        1_000,
        2_500,
        5_000,
        7_500,
        10_000,
    )

    # Root of the artefact tree. Every trainer appends its own algorithm key,
    # then the seed and policy directories, via src/training/model_paths.py, so
    # that a default run lands where the evaluators look.
    model_root_directory: str = "results/models"
    gamma: float = 1.0

    @property
    def default_seed(self) -> int:
        """First seed of the configured set.

        Used by the single-run training scripts, which train one model rather
        than a suite. Keeping it derived means it cannot drift away from
        ``seeds``.
        """
        return self.seeds[0]

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be greater than zero")

        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in range [0, 1]")

        if not self.seeds:
            raise ValueError("seeds must not be empty")

        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be unique: {self.seeds}")

        if any(seed < 0 for seed in self.seeds):
            raise ValueError(f"seeds must be non-negative: {self.seeds}")

        beyond_budget = [
            episode
            for episode in self.checkpoint_episodes
            if episode > self.episodes
        ]
        if beyond_budget:
            raise ValueError(
                "checkpoint_episodes must not exceed the training budget "
                f"({self.episodes}): {sorted(beyond_budget)}"
            )


# The full experiment. These are the numbers the thesis reports.
#
# Measured cost at ~83 ms/episode: 10 000 episodes x 5 seeds x 4 policies x
# 4 algorithms is roughly 18 hours on a single core.
FINAL_CONFIG = TrainingConfig()

# A cheap end-to-end rehearsal: same code path, same artefact layout, small
# enough to run before committing to the real thing. Roughly 33 minutes on a
# single core, about 4% of the full run.
#
# Three seeds rather than one because seed-level statistics need at least two,
# and a rehearsal that cannot produce a confidence interval would not exercise
# the reporting layer it is meant to rehearse.
VERIFICATION_CONFIG = replace(
    TrainingConfig(),
    episodes=500,
    seeds=(42, 123, 456),
    checkpoint_episodes=(100, 250, 500),
)

TRAINING_CONFIG_PRESETS = {
    "final": FINAL_CONFIG,
    "verification": VERIFICATION_CONFIG,
    "extended": FINAL_CONFIG,
}

DEFAULT_TRAINING_CONFIG_PRESET = "final"


def training_config_for(preset: str) -> TrainingConfig:
    try:
        return TRAINING_CONFIG_PRESETS[preset]
    except KeyError as error:
        raise ValueError(
            f"Unsupported training config preset: {preset!r}. "
            f"Choose one of {sorted(TRAINING_CONFIG_PRESETS)}."
        ) from error
