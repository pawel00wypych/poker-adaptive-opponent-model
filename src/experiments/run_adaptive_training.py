from collections import Counter
from time import perf_counter

from PyPokerEngine.pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.players.adaptive_player import AdaptivePlayer
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.config import GameConfig, TrainingConfig
from src.experiments.cli_utils import parse_training_args
from src.experiments.training_opponents import (
    build_training_opponent,
)


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    return (
        f"{int(hours):02d}:"
        f"{int(minutes):02d}:"
        f"{remaining_seconds:06.3f}"
    )


def run_adaptive_training(
    progress: bool = True,
    player_verbose: bool = False,
    player_log_interval: int = 1,
    engine_verbose: bool = False,
    log_interval: int = 100,
) -> None:
    game_config = GameConfig()
    training_config = TrainingConfig()

    agent = MonteCarloAgent(
        alpha=training_config.alpha,
        epsilon=training_config.epsilon_start,
        epsilon_min=training_config.epsilon_min,
    )
    agent.train()

    opponent_counter = Counter()

    training_start = perf_counter()

    for episode in range(training_config.episodes):
        episode_start = perf_counter()

        config = setup_config(
            max_round=game_config.max_round,
            initial_stack=game_config.initial_stack,
            small_blind_amount=game_config.small_blind_amount,
        )

        config.register_player(
            name="adaptive_mc",
            algorithm=AdaptivePlayer(
                agent=agent,
                player_name="adaptive_mc",
                verbose=player_verbose,
                log_interval=player_log_interval,
            ),
        )

        opponent_name, opponent = build_training_opponent(
            episode
        )
        opponent_counter[opponent_name] += 1

        config.register_player(
            name=opponent_name,
            algorithm=opponent,
        )

        start_poker(
            config,
            verbose=1 if engine_verbose else 0,
        )

        episode_duration = perf_counter() - episode_start

        if progress and (episode + 1) % log_interval == 0:
            elapsed = perf_counter() - training_start
            completed = episode + 1
            average_episode_time = elapsed / completed

            remaining_episodes = (
                training_config.episodes - completed
            )
            estimated_remaining = (
                average_episode_time * remaining_episodes
            )

            print(
                f"Adaptive episode "
                f"{completed}/{training_config.episodes}, "
                f"opponent={opponent_name}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"states={len(agent.q_table)}, "
                f"episode_time={episode_duration:.3f}s, "
                f"elapsed={format_duration(elapsed)}, "
                f"estimated_remaining="
                f"{format_duration(estimated_remaining)}"
            )

    training_duration = perf_counter() - training_start

    agent.save(training_config.adaptive_model_path)

    print(
        "Adaptive training finished\n"
        f"episodes={training_config.episodes}\n"
        f"duration={format_duration(training_duration)}\n"
        f"duration_seconds={training_duration:.3f}\n"
        f"average_episode_seconds="
        f"{training_duration / training_config.episodes:.6f}\n"
        f"states={len(agent.q_table)}\n"
        f"opponents={dict(opponent_counter)}\n"
        f"model={training_config.adaptive_model_path}"
    )


if __name__ == "__main__":
    args = parse_training_args()

    run_adaptive_training(
        progress=args.progress,
        player_verbose=args.player_verbose,
        player_log_interval=args.player_log_interval,
        engine_verbose=args.engine_verbose,
        log_interval=args.log_interval,
    )