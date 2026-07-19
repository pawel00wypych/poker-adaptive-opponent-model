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


@dataclass(frozen=True)
class TrainingConfig:
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

    single_policy_model_path: str = (
        "results/models/monte_carlo_single_policy.pkl"
    )

    fish_model_path: str = (
        "results/models/monte_carlo_vs_fish.pkl"
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



@dataclass(frozen=True)
class EvaluationConfig:
    games_per_matchup: int = 200
    output_path: str = "results/raw/agent_comparison_results.csv"
