from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Iterable

from pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.config import GameConfig, TrainingConfig
from src.experiments.constants import MODEL_TYPE_GENERAL_POLICY
from src.experiments.cli_utils import parse_training_args
from src.experiments.training_opponents import (
    build_training_opponent,
)
from src.players.general_policy_player import (
    GeneralPolicyPlayer,
)
from src.training.checkpoint_utils import (
    build_checkpoint_episodes,
    build_checkpoint_path,
)
from src.training.epsilon_schedule import calculate_epsilon
from src.training.random_utils import set_global_seed
from src.training.training_metadata import save_json


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    return (
        f"{int(hours):02d}:"
        f"{int(minutes):02d}:"
        f"{remaining_seconds:06.3f}"
    )


def build_metadata(
    completed_episodes: int,
    total_episodes: int,
    seed: int,
    epsilon_schedule: str,
    alpha_mode: str,
    agent: MonteCarloAgent,
    game_config: GameConfig,
    duration_seconds: float,
    total_hands: int,
    opponent_counter: Counter,
) -> dict:
    return {
        "model_type": MODEL_TYPE_GENERAL_POLICY,
        "opponent_type": "mixed",
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
        "opponents": dict(opponent_counter),
        "max_round": game_config.max_round,
        "initial_stack": game_config.initial_stack,
        "small_blind_amount": (
            game_config.small_blind_amount
        ),
        "duration_seconds": duration_seconds,
    }


def run_general_policy_training(
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

    training_seed = (
        seed
        if seed is not None
        else training_config.default_seed
    )

    selected_schedule = (
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
        else training_config.general_policy_model_path
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

    set_global_seed(training_seed)

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
            checkpoint_interval=checkpoint_interval,
        )
        if checkpoints_enabled
        else set()
    )

    opponent_counter = Counter()
    total_hands = 0
    training_start = perf_counter()

    for episode_index in range(total_episodes):
        epsilon = calculate_epsilon(
            schedule=selected_schedule,
            episode=episode_index,
            total_episodes=total_episodes,
            epsilon_start=training_config.epsilon_start,
            epsilon_min=training_config.epsilon_min,
        )

        agent.set_epsilon(epsilon)

        episode_start = perf_counter()

        config = setup_config(
            max_round=game_config.max_round,
            initial_stack=game_config.initial_stack,
            small_blind_amount=(
                game_config.small_blind_amount
            ),
        )

        player = GeneralPolicyPlayer(
            agent=agent,
            player_name="policy_general_mc",
            verbose=player_verbose,
            log_interval=player_log_interval,
        )

        opponent_name, opponent = (
            build_training_opponent(episode_index)
        )

        opponent_counter[opponent_name] += 1

        config.register_player(
            name="policy_general_mc",
            algorithm=player,
        )

        config.register_player(
            name=opponent_name,
            algorithm=opponent,
        )

        start_poker(
            config,
            verbose=1 if engine_verbose else 0,
        )

        total_hands += player.hands_played
        completed_episodes = episode_index + 1
        elapsed = perf_counter() - training_start

        if (
            progress
            and completed_episodes % log_interval == 0
        ):
            average_episode_time = (
                elapsed / completed_episodes
            )

            remaining = (
                total_episodes - completed_episodes
            )

            estimated_remaining = (
                average_episode_time * remaining
            )

            print(
                f"General-policy: "
                f"episode={completed_episodes}/"
                f"{total_episodes}, "
                f"seed={training_seed}, "
                f"opponent={opponent_name}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"states={len(agent.q_table)}, "
                f"episode_time="
                f"{perf_counter() - episode_start:.3f}s, "
                f"elapsed={format_duration(elapsed)}, "
                f"estimated_remaining="
                f"{format_duration(estimated_remaining)}"
            )

        if (
            checkpoints_enabled
            and completed_episodes in checkpoints
        ):
            checkpoint_path = build_checkpoint_path(
                checkpoint_directory=(
                    selected_checkpoint_directory
                ),
                model_name=MODEL_TYPE_GENERAL_POLICY,
                completed_episodes=completed_episodes,
                seed=training_seed,
            )

            metadata = build_metadata(
                completed_episodes=completed_episodes,
                total_episodes=total_episodes,
                seed=training_seed,
                epsilon_schedule=selected_schedule,
                alpha_mode=selected_alpha_mode,
                agent=agent,
                game_config=game_config,
                duration_seconds=elapsed,
                total_hands=total_hands,
                opponent_counter=opponent_counter,
            )

            agent.save(
                checkpoint_path,
                metadata=metadata,
            )

    training_duration = (
        perf_counter() - training_start
    )

    final_metadata = build_metadata(
        completed_episodes=total_episodes,
        total_episodes=total_episodes,
        seed=training_seed,
        epsilon_schedule=selected_schedule,
        alpha_mode=selected_alpha_mode,
        agent=agent,
        game_config=game_config,
        duration_seconds=training_duration,
        total_hands=total_hands,
        opponent_counter=opponent_counter,
    )

    agent.save(
        final_model_path,
        metadata=final_metadata,
    )

    metadata_path = (
        Path(final_model_path).with_suffix(".json")
    )

    save_json(
        metadata_path,
        final_metadata,
    )

    print(
        "General-policy training finished\n"
        f"episodes={total_episodes}\n"
        f"seed={training_seed}\n"
        f"epsilon_schedule={selected_schedule}\n"
        f"alpha_mode={selected_alpha_mode}\n"
        f"final_epsilon={agent.epsilon:.6f}\n"
        f"duration={format_duration(training_duration)}\n"
        f"duration_seconds={training_duration:.3f}\n"
        f"total_hands={total_hands}\n"
        f"states={len(agent.q_table)}\n"
        f"opponents={dict(opponent_counter)}\n"
        f"model={final_model_path}"
    )

    return final_metadata


if __name__ == "__main__":
    args = parse_training_args()

    run_general_policy_training(
        episodes=args.episodes,
        seed=args.seed,
        epsilon_schedule=args.epsilon_schedule,
        alpha_mode=args.alpha_mode,
        output_path=args.output_path,
        checkpoint_directory=(
            args.checkpoint_directory
        ),
        checkpoint_episodes=(
            args.checkpoint_episodes
        ),
        checkpoints_enabled=args.checkpoints,
        checkpoint_interval=(
            args.checkpoint_interval
        ),
        progress=args.progress,
        player_verbose=args.player_verbose,
        player_log_interval=(
            args.player_log_interval
        ),
        engine_verbose=args.engine_verbose,
        log_interval=args.log_interval,
    )
