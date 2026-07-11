from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    max_round: int = 100
    initial_stack: int = 100
    small_blind_amount: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 5_000
    alpha: float = 0.1
    epsilon_start: float = 0.5
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    adaptive_model_path: str = (
        "results/models/monte_carlo_adaptive.pkl"
    )

    single_policy_model_path: str = (
        "results/models/monte_carlo_single_policy.pkl"
    )


@dataclass(frozen=True)
class EvaluationConfig:
    games: int = 120
    max_round: int = 100
    output_path: str = "results/raw/agent_comparison_results.csv"