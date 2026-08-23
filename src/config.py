from dataclasses import dataclass

from src.training.constants import (
    ALPHA_MODE_CONSTANT,
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
    # Every algorithm shares this budget so that comparisons measure the
    # algorithm rather than how many episodes each one happened to get.
    episodes: int = 7_500
    alpha: float = 0.1
    alpha_mode: str = ALPHA_MODE_CONSTANT

    epsilon_start: float = 0.5
    epsilon_min: float = 0.05
    epsilon_schedule: str = EPSILON_SCHEDULE_LINEAR

    default_seed: int = 42

    checkpoint_episodes: tuple[int, ...] = (
        1_000,
        2_500,
        5_000,
        7_500,
    )

    general_policy_model_path: str = (
        "results/models/monte_carlo_general_policy.pkl"
    )

    tight_model_path: str = (
        "results/models/monte_carlo_vs_tight.pkl"
    )

    aggressive_model_path: str = (
        "results/models/monte_carlo_vs_aggressive.pkl"
    )

    calling_model_path: str = (
        "results/models/checkpoints/monte_carlo_vs_calling_episodes_7500_seed_42.pkl"
    )

    checkpoint_directory: str = (
        "results/models/checkpoints"
    )
