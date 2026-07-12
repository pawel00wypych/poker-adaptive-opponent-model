from time import perf_counter

from PyPokerEngine.pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.players.specialist_policy_player import (
    SpecialistPolicyPlayer,
)
from src.config import GameConfig, TrainingConfig
from src.experiments.cli_utils import (
    parse_specialist_training_args,
)
from src.experiments.training_opponents import (
    build_opponent,
)


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(
        seconds,
        3600,
    )
    minutes, remaining_seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{int(hours):02d}:"
        f"{int(minutes):02d}:"
        f"{remaining_seconds:06.3f}"
    )


def get_model_path(
    training_config: TrainingConfig,
    opponent_type: str,
) -> str:
    paths = {
        "fish": training_config.fish_model_path,
        "aggressive": (
            training_config.aggressive_model_path
        ),
        "calling": (
            training_config.calling_model_path
        ),
    }

    return paths[opponent_type]


def run_specialist_training(
    opponent_type: str,
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
        epsilon_decay=training_config.epsilon_decay,
    )
    agent.train()

    training_start = perf_counter()

    for episode in range(
        training_config.episodes
    ):
        episode_start = perf_counter()

        config = setup_config(
            max_round=game_config.max_round,
            initial_stack=game_config.initial_stack,
            small_blind_amount=(
                game_config.small_blind_amount
            ),
        )

        specialist_player = SpecialistPolicyPlayer(
            agent=agent,
            opponent_type=opponent_type,
            player_name="specialist_mc",
            verbose=player_verbose,
            log_interval=player_log_interval,
        )

        opponent = build_opponent(
            opponent_type
        )

        config.register_player(
            name="specialist_mc",
            algorithm=specialist_player,
        )

        config.register_player(
            name=opponent_type,
            algorithm=opponent,
        )

        start_poker(
            config,
            verbose=1 if engine_verbose else 0,
        )

        episode_duration = (
            perf_counter() - episode_start
        )

        if (
            progress
            and (episode + 1) % log_interval == 0
        ):
            elapsed = (
                perf_counter() - training_start
            )
            completed = episode + 1
            average_episode_time = (
                elapsed / completed
            )

            remaining_episodes = (
                training_config.episodes
                - completed
            )

            estimated_remaining = (
                average_episode_time
                * remaining_episodes
            )

            print(
                f"Specialist vs {opponent_type}: "
                f"episode "
                f"{completed}/"
                f"{training_config.episodes}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"states={len(agent.q_table)}, "
                f"episode_time="
                f"{episode_duration:.3f}s, "
                f"elapsed="
                f"{format_duration(elapsed)}, "
                f"estimated_remaining="
                f"{format_duration(estimated_remaining)}"
            )

    training_duration = (
        perf_counter() - training_start
    )

    model_path = get_model_path(
        training_config,
        opponent_type,
    )

    agent.save(model_path)

    print(
        "Specialist training finished\n"
        f"opponent={opponent_type}\n"
        f"episodes="
        f"{training_config.episodes}\n"
        f"duration="
        f"{format_duration(training_duration)}\n"
        f"duration_seconds="
        f"{training_duration:.3f}\n"
        f"average_episode_seconds="
        f"{training_duration / training_config.episodes:.6f}\n"
        f"states={len(agent.q_table)}\n"
        f"model={model_path}"
    )


if __name__ == "__main__":
    args = parse_specialist_training_args()

    run_specialist_training(
        opponent_type=args.opponent,
        progress=args.progress,
        player_verbose=args.player_verbose,
        player_log_interval=(
            args.player_log_interval
        ),
        engine_verbose=args.engine_verbose,
        log_interval=args.log_interval,
    )