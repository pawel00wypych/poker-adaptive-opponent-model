from pathlib import Path
from time import perf_counter
from typing import Iterable

from pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.players.specialist_policy_player import (
    SpecialistPolicyPlayer,
)
from src.config import GameConfig, TrainingConfig
from src.experiments.constants import MODEL_TYPE_SPECIALIST
from src.experiments.cli_utils import (
    parse_specialist_training_args,
)
from src.experiments.training_opponents import (
    build_opponent,
)
from src.training.checkpoint_utils import (
    build_checkpoint_episodes,
    build_checkpoint_path,
)
from src.training.epsilon_schedule import (
    calculate_epsilon,
)
from src.training.random_utils import (
    set_global_seed,
)
from src.training.training_metadata import (
    save_json,
)
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
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


def get_default_model_path(
    training_config: TrainingConfig,
    opponent_type: str,
) -> str:
    model_paths = {
        OPPONENT_TYPE_TIGHT: training_config.tight_model_path,
        OPPONENT_TYPE_AGGRESSIVE: (
            training_config.aggressive_model_path
        ),
        OPPONENT_TYPE_CALLING: (
            training_config.calling_model_path
        ),
    }

    try:
        return model_paths[opponent_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported opponent type: {opponent_type}"
        ) from error


def build_training_metadata(
    opponent_type: str,
    completed_episodes: int,
    total_episodes: int,
    seed: int,
    epsilon_schedule: str,
    alpha_mode: str,
    agent: MonteCarloAgent,
    game_config: GameConfig,
    duration_seconds: float,
    total_hands: int,
) -> dict:
    mean_hands_per_episode = (
        total_hands / completed_episodes
        if completed_episodes > 0
        else 0.0
    )

    hands_per_second = (
        total_hands / duration_seconds
        if duration_seconds > 0
        else 0.0
    )

    return {
        "model_type": MODEL_TYPE_SPECIALIST,
        "opponent_type": opponent_type,
        "completed_episodes": completed_episodes,
        "total_planned_episodes": total_episodes,
        "seed": seed,
        "epsilon_schedule": epsilon_schedule,
        "alpha_mode": alpha_mode,
        "current_epsilon": agent.epsilon,
        "alpha": agent.alpha,
        "epsilon_min": agent.epsilon_min,
        "states": len(agent.q_table),
        "total_hands": total_hands,
        "mean_hands_per_episode": (
            mean_hands_per_episode
        ),
        "hands_per_second": hands_per_second,
        "max_round": game_config.max_round,
        "initial_stack": game_config.initial_stack,
        "small_blind_amount": (
            game_config.small_blind_amount
        ),
        "duration_seconds": duration_seconds,
    }


def run_specialist_training(
    opponent_type: str,
    episodes: int | None = None,
    seed: int | None = None,
    epsilon_schedule: str | None = None,
    alpha_mode: str | None = None,
    output_path: str | None = None,
    checkpoint_directory: str | None = None,
    checkpoint_episodes: Iterable[int] | None = None,
    checkpoints_enabled: bool = True,
    checkpoint_interval: int | None = None,
    progress: bool = True,
    player_verbose: bool = False,
    player_log_interval: int = 1,
    engine_verbose: bool = False,
    log_interval: int = 100,
) -> dict:
    game_config = GameConfig()
    training_config = TrainingConfig()

    total_episodes = (
        episodes
        if episodes is not None
        else training_config.episodes
    )

    if total_episodes <= 0:
        raise ValueError(
            "episodes must be greater than zero"
        )

    training_seed = (
        seed
        if seed is not None
        else training_config.default_seed
    )

    if training_seed < 0:
        raise ValueError(
            "seed must be non-negative"
        )

    selected_epsilon_schedule = (
        epsilon_schedule
        if epsilon_schedule is not None
        else training_config.epsilon_schedule
    )

    selected_alpha_mode = (
        alpha_mode
        if alpha_mode is not None
        else training_config.alpha_mode
    )

    final_model_path = (
        output_path
        if output_path is not None
        else get_default_model_path(
            training_config,
            opponent_type,
        )
    )

    selected_checkpoint_directory = (
        checkpoint_directory
        if checkpoint_directory is not None
        else training_config.checkpoint_directory
    )

    selected_checkpoint_episodes = tuple(
        checkpoint_episodes
        if checkpoint_episodes is not None
        else training_config.checkpoint_episodes
    )

    set_global_seed(
        training_seed
    )

    agent = MonteCarloAgent(
        alpha=training_config.alpha,
        epsilon=training_config.epsilon_start,
        epsilon_min=training_config.epsilon_min,
        alpha_mode=selected_alpha_mode,
    )
    agent.train()

    checkpoints = (
        build_checkpoint_episodes(
            total_episodes=total_episodes,
            configured_checkpoints=(
                selected_checkpoint_episodes
            ),
            checkpoint_interval=(
                checkpoint_interval
            ),
        )
        if checkpoints_enabled
        else set()
    )

    training_start = perf_counter()
    total_hands = 0

    for episode_index in range(
        total_episodes
    ):
        epsilon = calculate_epsilon(
            schedule=selected_epsilon_schedule,
            episode=episode_index,
            total_episodes=total_episodes,
            epsilon_start=(
                training_config.epsilon_start
            ),
            epsilon_min=(
                training_config.epsilon_min
            ),
        )

        agent.set_epsilon(
            epsilon
        )

        episode_start = perf_counter()

        config = setup_config(
            max_round=game_config.max_round,
            initial_stack=(
                game_config.initial_stack
            ),
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

        completed_episodes = (
            episode_index + 1
        )

        total_hands += (
            specialist_player.hands_played
        )

        episode_duration = (
            perf_counter() - episode_start
        )

        elapsed = (
            perf_counter() - training_start
        )

        if (
            progress
            and completed_episodes
            % log_interval
            == 0
        ):
            average_episode_time = (
                elapsed / completed_episodes
            )

            remaining_episodes = (
                total_episodes
                - completed_episodes
            )

            estimated_remaining = (
                average_episode_time
                * remaining_episodes
            )

            mean_hands_per_episode = (
                total_hands
                / completed_episodes
            )

            hands_per_second = (
                total_hands / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"Specialist vs {opponent_type}: "
                f"episode={completed_episodes}/"
                f"{total_episodes}, "
                f"seed={training_seed}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"schedule="
                f"{selected_epsilon_schedule}, "
                f"states={len(agent.q_table)}, "
                f"episode_time="
                f"{episode_duration:.3f}s, "
                f"total_hands={total_hands}, "
                f"mean_hands="
                f"{mean_hands_per_episode:.2f}, "
                f"hands_per_second="
                f"{hands_per_second:.2f}, "
                f"elapsed="
                f"{format_duration(elapsed)}, "
                f"estimated_remaining="
                f"{format_duration(estimated_remaining)}"
            )

        if (
            checkpoints_enabled
            and completed_episodes
            in checkpoints
        ):
            checkpoint_path = (
                build_checkpoint_path(
                    checkpoint_directory=(
                        selected_checkpoint_directory
                    ),
                    model_name=(
                        f"specialist_{opponent_type}"
                    ),
                    completed_episodes=(
                        completed_episodes
                    ),
                    seed=training_seed,
                )
            )

            checkpoint_metadata = (
                build_training_metadata(
                    opponent_type=opponent_type,
                    completed_episodes=(
                        completed_episodes
                    ),
                    total_episodes=(
                        total_episodes
                    ),
                    seed=training_seed,
                    epsilon_schedule=(
                        selected_epsilon_schedule
                    ),
                    alpha_mode=selected_alpha_mode,
                    agent=agent,
                    game_config=game_config,
                    duration_seconds=elapsed,
                    total_hands=total_hands,
                )
            )

            agent.save(
                checkpoint_path,
                metadata=checkpoint_metadata,
            )

            checkpoint_metadata_path = (
                Path(checkpoint_path)
                .with_suffix(".json")
            )

            save_json(
                checkpoint_metadata_path,
                checkpoint_metadata,
            )

            if progress:
                print(
                    "Saved checkpoint: "
                    f"{checkpoint_path}"
                )

    training_duration = (
        perf_counter() - training_start
    )

    final_metadata = (
        build_training_metadata(
            opponent_type=opponent_type,
            completed_episodes=total_episodes,
            total_episodes=total_episodes,
            seed=training_seed,
            epsilon_schedule=(
                selected_epsilon_schedule
            ),
            alpha_mode=selected_alpha_mode,
            agent=agent,
            game_config=game_config,
            duration_seconds=(
                training_duration
            ),
            total_hands=total_hands,
        )
    )

    agent.save(
        final_model_path,
        metadata=final_metadata,
    )

    final_metadata_path = (
        Path(final_model_path)
        .with_suffix(".json")
    )

    save_json(
        final_metadata_path,
        final_metadata,
    )

    average_episode_seconds = (
        training_duration
        / total_episodes
    )

    mean_hands_per_episode = (
        total_hands
        / total_episodes
    )

    hands_per_second = (
        total_hands / training_duration
        if training_duration > 0
        else 0.0
    )

    print(
        "Specialist training finished\n"
        f"opponent={opponent_type}\n"
        f"episodes={total_episodes}\n"
        f"seed={training_seed}\n"
        f"epsilon_schedule="
        f"{selected_epsilon_schedule}\n"
        f"alpha_mode={selected_alpha_mode}\n"
        f"final_epsilon="
        f"{agent.epsilon:.6f}\n"
        f"duration="
        f"{format_duration(training_duration)}\n"
        f"duration_seconds="
        f"{training_duration:.3f}\n"
        f"average_episode_seconds="
        f"{average_episode_seconds:.6f}\n"
        f"total_hands={total_hands}\n"
        f"mean_hands_per_episode="
        f"{mean_hands_per_episode:.3f}\n"
        f"hands_per_second="
        f"{hands_per_second:.3f}\n"
        f"states={len(agent.q_table)}\n"
        f"model={final_model_path}\n"
        f"metadata={final_metadata_path}"
    )

    return final_metadata


if __name__ == "__main__":
    args = parse_specialist_training_args()

    run_specialist_training(
        opponent_type=args.opponent,
        episodes=args.episodes,
        seed=args.seed,
        epsilon_schedule=(
            args.epsilon_schedule
        ),
        alpha_mode=args.alpha_mode,
        output_path=args.output_path,
        checkpoint_directory=(
            args.checkpoint_directory
        ),
        checkpoint_episodes=(
            args.checkpoint_episodes
        ),
        checkpoints_enabled=(
            args.checkpoints
        ),
        checkpoint_interval=(
            args.checkpoint_interval
        ),
        progress=args.progress,
        player_verbose=(
            args.player_verbose
        ),
        player_log_interval=(
            args.player_log_interval
        ),
        engine_verbose=(
            args.engine_verbose
        ),
        log_interval=args.log_interval,
    )
