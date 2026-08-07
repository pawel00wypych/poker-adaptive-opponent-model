from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Iterable

from PyPokerEngine.pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.agents.sarsa_agent import SarsaAgent
from src.config import GameConfig, TrainingConfig
from src.experiments.constants import (
    MODEL_TYPE_SINGLE_POLICY,
    MODEL_TYPE_SPECIALIST,
)
from src.experiments.training_opponents import (
    build_opponent,
    build_training_opponent,
)
from src.players.single_policy_player import SinglePolicyPlayer
from src.players.specialist_policy_player import SpecialistPolicyPlayer
from src.poker.constants import TRAINING_OPPONENT_TYPES
from src.training.checkpoint_utils import (
    build_checkpoint_episodes,
    build_checkpoint_path,
)
from src.training.epsilon_schedule import calculate_epsilon
from src.training.random_utils import set_global_seed
from src.training.training_metadata import save_json

SARSA_ALGORITHM_NAME = "sarsa"


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    return (
        f"{int(hours):02d}:"
        f"{int(minutes):02d}:"
        f"{remaining_seconds:06.3f}"
    )


def model_run_name(model_type: str) -> str:
    if model_type == MODEL_TYPE_SINGLE_POLICY:
        return MODEL_TYPE_SINGLE_POLICY

    if model_type in TRAINING_OPPONENT_TYPES:
        return f"specialist_{model_type}"

    raise ValueError(
        f"Unsupported SARSA model type: {model_type}"
    )


def default_model_path(model_type: str) -> str:
    return str(
        Path("results/models/sarsa")
        / model_run_name(model_type)
        / "final.pkl"
    )


def default_checkpoint_directory(model_type: str) -> str:
    return str(
        Path("results/models/sarsa")
        / model_run_name(model_type)
        / "checkpoints"
    )


def checkpoint_model_name(model_type: str) -> str:
    return model_run_name(model_type)


def build_sarsa_metadata(
    *,
    model_type: str,
    opponent_type: str,
    completed_episodes: int,
    total_episodes: int,
    seed: int,
    epsilon_schedule: str,
    agent: SarsaAgent,
    game_config: GameConfig,
    duration_seconds: float,
    total_hands: int,
    opponent_counter: Counter | None = None,
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

    metadata = {
        "algorithm": SARSA_ALGORITHM_NAME,
        "model_type": model_type,
        "opponent_type": opponent_type,
        "completed_episodes": completed_episodes,
        "total_planned_episodes": total_episodes,
        "seed": seed,
        "epsilon_schedule": epsilon_schedule,
        "current_epsilon": agent.epsilon,
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "epsilon_min": agent.epsilon_min,
        "states": len(agent.q_table),
        "total_hands": total_hands,
        "mean_hands_per_episode": mean_hands_per_episode,
        "hands_per_second": hands_per_second,
        "max_round": game_config.max_round,
        "initial_stack": game_config.initial_stack,
        "small_blind_amount": game_config.small_blind_amount,
        "duration_seconds": duration_seconds,
    }

    if opponent_counter is not None:
        metadata["opponents"] = dict(opponent_counter)

    return metadata


def build_training_player(
    *,
    model_type: str,
    agent: SarsaAgent,
    player_verbose: bool,
    player_log_interval: int,
):
    if model_type == MODEL_TYPE_SINGLE_POLICY:
        return SinglePolicyPlayer(
            agent=agent,
            player_name="single_policy_sarsa",
            verbose=player_verbose,
            log_interval=player_log_interval,
        )

    if model_type in TRAINING_OPPONENT_TYPES:
        return SpecialistPolicyPlayer(
            agent=agent,
            opponent_type=model_type,
            player_name="specialist_sarsa",
            verbose=player_verbose,
            log_interval=player_log_interval,
        )

    raise ValueError(
        f"Unsupported SARSA model type: {model_type}"
    )


def build_episode_opponent(
    *,
    model_type: str,
    episode_index: int,
):
    if model_type == MODEL_TYPE_SINGLE_POLICY:
        return build_training_opponent(episode_index)

    if model_type in TRAINING_OPPONENT_TYPES:
        return model_type, build_opponent(model_type)

    raise ValueError(
        f"Unsupported SARSA model type: {model_type}"
    )


def run_sarsa_model_training(
    *,
    model_type: str,
    episodes: int | None = None,
    seed: int | None = None,
    epsilon_schedule: str | None = None,
    alpha: float | None = None,
    gamma: float = 1.0,
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

    if player_log_interval <= 0:
        raise ValueError(
            "player_log_interval must be greater than zero"
        )

    if log_interval <= 0:
        raise ValueError(
            "log_interval must be greater than zero"
        )

    selected_schedule = (
        epsilon_schedule
        if epsilon_schedule is not None
        else training_config.epsilon_schedule
    )

    selected_alpha = (
        alpha
        if alpha is not None
        else training_config.alpha
    )

    final_model_path = (
        output_path
        if output_path is not None
        else default_model_path(model_type)
    )

    selected_checkpoint_directory = (
        checkpoint_directory
        if checkpoint_directory is not None
        else default_checkpoint_directory(model_type)
    )

    selected_checkpoint_episodes = tuple(
        checkpoint_episodes
        if checkpoint_episodes is not None
        else training_config.checkpoint_episodes
    )

    set_global_seed(training_seed)

    agent = SarsaAgent(
        alpha=selected_alpha,
        gamma=gamma,
        epsilon=training_config.epsilon_start,
        epsilon_min=training_config.epsilon_min,
    )
    agent.train()

    checkpoints = (
        build_checkpoint_episodes(
            total_episodes=total_episodes,
            configured_checkpoints=selected_checkpoint_episodes,
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
            small_blind_amount=game_config.small_blind_amount,
        )

        player = build_training_player(
            model_type=model_type,
            agent=agent,
            player_verbose=player_verbose,
            player_log_interval=player_log_interval,
        )

        opponent_name, opponent = build_episode_opponent(
            model_type=model_type,
            episode_index=episode_index,
        )
        opponent_counter[opponent_name] += 1

        config.register_player(
            name="sarsa",
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
            average_episode_time = elapsed / completed_episodes
            remaining = total_episodes - completed_episodes
            estimated_remaining = average_episode_time * remaining

            print(
                f"SARSA {model_type}: "
                f"episode={completed_episodes}/"
                f"{total_episodes}, "
                f"seed={training_seed}, "
                f"opponent={opponent_name}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"gamma={agent.gamma:.4f}, "
                f"states={len(agent.q_table)}, "
                f"episode_time={perf_counter() - episode_start:.3f}s, "
                f"elapsed={format_duration(elapsed)}, "
                f"estimated_remaining="
                f"{format_duration(estimated_remaining)}"
            )

        if (
            checkpoints_enabled
            and completed_episodes in checkpoints
        ):
            checkpoint_path = build_checkpoint_path(
                checkpoint_directory=selected_checkpoint_directory,
                model_name=checkpoint_model_name(model_type),
                completed_episodes=completed_episodes,
                seed=training_seed,
            )

            metadata = build_sarsa_metadata(
                model_type=(
                    MODEL_TYPE_SINGLE_POLICY
                    if model_type == MODEL_TYPE_SINGLE_POLICY
                    else MODEL_TYPE_SPECIALIST
                ),
                opponent_type=(
                    "mixed"
                    if model_type == MODEL_TYPE_SINGLE_POLICY
                    else model_type
                ),
                completed_episodes=completed_episodes,
                total_episodes=total_episodes,
                seed=training_seed,
                epsilon_schedule=selected_schedule,
                agent=agent,
                game_config=game_config,
                duration_seconds=elapsed,
                total_hands=total_hands,
                opponent_counter=(
                    opponent_counter
                    if model_type == MODEL_TYPE_SINGLE_POLICY
                    else None
                ),
            )

            agent.save(
                checkpoint_path,
                metadata=metadata,
            )
            save_json(
                Path(checkpoint_path).with_suffix(".json"),
                metadata,
            )

            if progress:
                print(
                    "Saved SARSA checkpoint: "
                    f"{checkpoint_path}"
                )

    training_duration = perf_counter() - training_start

    final_metadata = build_sarsa_metadata(
        model_type=(
            MODEL_TYPE_SINGLE_POLICY
            if model_type == MODEL_TYPE_SINGLE_POLICY
            else MODEL_TYPE_SPECIALIST
        ),
        opponent_type=(
            "mixed"
            if model_type == MODEL_TYPE_SINGLE_POLICY
            else model_type
        ),
        completed_episodes=total_episodes,
        total_episodes=total_episodes,
        seed=training_seed,
        epsilon_schedule=selected_schedule,
        agent=agent,
        game_config=game_config,
        duration_seconds=training_duration,
        total_hands=total_hands,
        opponent_counter=(
            opponent_counter
            if model_type == MODEL_TYPE_SINGLE_POLICY
            else None
        ),
    )

    agent.save(
        final_model_path,
        metadata=final_metadata,
    )
    save_json(
        Path(final_model_path).with_suffix(".json"),
        final_metadata,
    )

    print(
        "SARSA training finished\n"
        f"model_type={model_type}\n"
        f"episodes={total_episodes}\n"
        f"seed={training_seed}\n"
        f"epsilon_schedule={selected_schedule}\n"
        f"gamma={agent.gamma:.6f}\n"
        f"final_epsilon={agent.epsilon:.6f}\n"
        f"duration={format_duration(training_duration)}\n"
        f"duration_seconds={training_duration:.3f}\n"
        f"total_hands={total_hands}\n"
        f"states={len(agent.q_table)}\n"
        f"opponents={dict(opponent_counter)}\n"
        f"model={final_model_path}"
    )

    return final_metadata
