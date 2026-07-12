from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    max_round: int = 100
    initial_stack: int = 200
    small_blind_amount: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 5_000
    alpha: float = 0.1
    epsilon_start: float = 0.5
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

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
        "results/models/monte_carlo_vs_calling.pkl"
    )



@dataclass(frozen=True)
class EvaluationConfig:
    games_per_matchup: int = 200
    output_path: str = "results/raw/agent_comparison_results.csv"