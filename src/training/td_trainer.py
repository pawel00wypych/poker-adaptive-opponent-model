import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Protocol

from pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.config import GameConfig, TrainingConfig
from src.players.learned.general_policy_player import GeneralPolicyPlayer
from src.players.learned.specialist_policy_player import SpecialistPolicyPlayer
from src.players.opponents.factory import (
    build_opponent,
    build_training_opponent,
)
from src.poker.constants import TRAINING_OPPONENT_TYPES
from src.rl.rng import (
    attach_rng,
    derive_episode_streams,
    seed_engine_stream,
)
from src.training.checkpoint_utils import (
    build_checkpoint_episodes,
    build_checkpoint_path,
)
from src.training.constants import (
    MODEL_TYPE_GENERAL_POLICY,
    MODEL_TYPE_SPECIALIST,
)
from src.training.epsilon_schedule import calculate_epsilon
from src.training.model_paths import (
    default_checkpoint_directory_path,
    default_final_model_path,
    policy_directory_name,
)
from src.training.random_utils import set_global_seed
from src.training.training_metadata import save_json


class TabularTDAgent(Protocol):
    alpha: float
    alpha_mode: str
    gamma: float
    epsilon: float
    epsilon_min: float
    q_table: dict

    def train(self) -> None: ...

    def set_epsilon(
        self,
        epsilon: float,
    ) -> None: ...

    def save(
        self,
        path: str | Path,
        metadata: dict | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class TDTrainingSpec:
    algorithm_name: str
    display_name: str
    model_root_directory: str
    agent_factory: Callable[..., TabularTDAgent]
    player_name_suffix: str
    registered_player_name: str


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    return (
        f"{int(hours):02d}:"
        f"{int(minutes):02d}:"
        f"{remaining_seconds:06.3f}"
    )


def model_run_name(
    model_type: str,
    *,
    error_label: str = "TD",
) -> str:
    return policy_directory_name(model_type, error_label=error_label)


def default_model_path(
    *,
    spec: TDTrainingSpec,
    model_type: str,
    seed: int,
) -> str:
    return default_final_model_path(
        model_root_directory=spec.model_root_directory,
        seed=seed,
        model_type=model_type,
        error_label=spec.display_name,
    )


def default_checkpoint_directory(
    *,
    spec: TDTrainingSpec,
    model_type: str,
    seed: int,
) -> str:
    return default_checkpoint_directory_path(
        model_root_directory=spec.model_root_directory,
        seed=seed,
        model_type=model_type,
        error_label=spec.display_name,
    )


def checkpoint_model_name(
    model_type: str,
    *,
    error_label: str = "TD",
) -> str:
    return model_run_name(
        model_type,
        error_label=error_label,
    )


def build_td_metadata(
    *,
    spec: TDTrainingSpec,
    model_type: str,
    opponent_type: str,
    completed_episodes: int,
    total_episodes: int,
    seed: int,
    epsilon_schedule: str,
    agent: TabularTDAgent,
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
        "algorithm": spec.algorithm_name,
        "model_type": model_type,
        "opponent_type": opponent_type,
        "completed_episodes": completed_episodes,
        "total_planned_episodes": total_episodes,
        "seed": seed,
        "epsilon_schedule": epsilon_schedule,
        "current_epsilon": agent.epsilon,
        "alpha": agent.alpha,
        "alpha_mode": agent.alpha_mode,
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
    spec: TDTrainingSpec,
    model_type: str,
    agent: TabularTDAgent,
    player_verbose: bool,
    player_log_interval: int,
):
    if model_type == MODEL_TYPE_GENERAL_POLICY:
        return GeneralPolicyPlayer(
            agent=agent,
            player_name=f"general_policy_{spec.player_name_suffix}",
            verbose=player_verbose,
            log_interval=player_log_interval,
        )

    if model_type in TRAINING_OPPONENT_TYPES:
        return SpecialistPolicyPlayer(
            agent=agent,
            opponent_type=model_type,
            player_name=f"specialist_{spec.player_name_suffix}",
            verbose=player_verbose,
            log_interval=player_log_interval,
        )

    raise ValueError(
        f"Unsupported {spec.display_name} model type: {model_type}"
    )


def build_episode_opponent(
    *,
    model_type: str,
    episode_index: int,
    error_label: str = "TD",
    rng: random.Random | None = None,
):
    if model_type == MODEL_TYPE_GENERAL_POLICY:
        return build_training_opponent(episode_index, rng=rng)

    if model_type in TRAINING_OPPONENT_TYPES:
        return model_type, build_opponent(model_type, rng=rng)

    raise ValueError(
        f"Unsupported {error_label} model type: {model_type}"
    )


def normalized_model_type(model_type: str) -> str:
    if model_type == MODEL_TYPE_GENERAL_POLICY:
        return MODEL_TYPE_GENERAL_POLICY

    return MODEL_TYPE_SPECIALIST


def normalized_opponent_type(model_type: str) -> str:
    if model_type == MODEL_TYPE_GENERAL_POLICY:
        return "mixed"

    return model_type


def run_td_model_training(
    *,
    spec: TDTrainingSpec,
    model_type: str,
    episodes: int | None = None,
    seed: int | None = None,
    epsilon_schedule: str | None = None,
    alpha: float | None = None,
    alpha_mode: str | None = None,
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

    selected_alpha_mode = (
        alpha_mode
        if alpha_mode is not None
        else training_config.alpha_mode
    )

    final_model_path = (
        output_path
        if output_path is not None
        else default_model_path(
            spec=spec,
            model_type=model_type,
            seed=training_seed,
        )
    )

    selected_checkpoint_directory = (
        checkpoint_directory
        if checkpoint_directory is not None
        else default_checkpoint_directory(
            spec=spec,
            model_type=model_type,
            seed=training_seed,
        )
    )

    selected_checkpoint_episodes = tuple(
        checkpoint_episodes
        if checkpoint_episodes is not None
        else training_config.checkpoint_episodes
    )

    set_global_seed(training_seed)

    agent = spec.agent_factory(
        alpha=selected_alpha,
        alpha_mode=selected_alpha_mode,
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

        streams = derive_episode_streams(training_seed, episode_index)
        seed_engine_stream(streams.deck_seed)

        config = setup_config(
            max_round=game_config.max_round,
            initial_stack=game_config.initial_stack,
            small_blind_amount=game_config.small_blind_amount,
        )

        player = build_training_player(
            spec=spec,
            model_type=model_type,
            agent=agent,
            player_verbose=player_verbose,
            player_log_interval=player_log_interval,
        )

        opponent_name, opponent = build_episode_opponent(
            model_type=model_type,
            episode_index=episode_index,
            error_label=spec.display_name,
            rng=streams.opponent,
        )
        opponent_counter[opponent_name] += 1

        attach_rng(player, streams.agent)
        attach_rng(opponent, streams.opponent)

        config.register_player(
            name=spec.registered_player_name,
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
                f"{spec.display_name} {model_type}: "
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
                model_name=checkpoint_model_name(
                    model_type,
                    error_label=spec.display_name,
                ),
                completed_episodes=completed_episodes,
                seed=training_seed,
            )

            metadata = build_td_metadata(
                spec=spec,
                model_type=normalized_model_type(model_type),
                opponent_type=normalized_opponent_type(model_type),
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
                    if model_type == MODEL_TYPE_GENERAL_POLICY
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
                    f"Saved {spec.display_name} checkpoint: "
                    f"{checkpoint_path}"
                )

    training_duration = perf_counter() - training_start

    final_metadata = build_td_metadata(
        spec=spec,
        model_type=normalized_model_type(model_type),
        opponent_type=normalized_opponent_type(model_type),
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
            if model_type == MODEL_TYPE_GENERAL_POLICY
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
        f"{spec.display_name} training finished\n"
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
