from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    max_round: int = 100
    initial_stack: int = 100
    small_blind_amount: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 2_000
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 0.5
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    model_path: str = "results/models/q_agent.pkl"


@dataclass(frozen=True)
class EvaluationConfig:
    games: int = 30
    max_round: int = 100
    model_path: str = "results/models/q_agent.pkl"
    output_path: str = "results/raw/evaluation_results.csv"