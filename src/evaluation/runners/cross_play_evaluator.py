import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.config import GameConfig
from src.evaluation.constants import (
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
)
from src.evaluation.player_factory import (
    EvaluationAgentLoaders,
    build_evaluation_player,
)
from src.evaluation.runners.checkpoint_evaluator import (
    CHECKPOINT_EVALUATION_FIELDNAMES,
    ModelBundle,
    build_result_row,
    get_classifier_metrics,
    get_hands_played,
    load_adaptive_agents,
    load_double_q_learning_adaptive_agents,
    load_double_q_learning_eval_agent,
    load_eval_agent,
    load_q_learning_adaptive_agents,
    load_q_learning_eval_agent,
    load_sarsa_adaptive_agents,
    load_sarsa_eval_agent,
)


CROSS_PLAY_EVALUATION_TYPE = "cross_play"

ADAPTIVE_CROSS_PLAY_AGENTS = (
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
)

POLICY_GENERAL_CROSS_PLAY_AGENTS = (
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
)

SUPPORTED_CROSS_PLAY_AGENTS = set(ADAPTIVE_CROSS_PLAY_AGENTS) | set(
    POLICY_GENERAL_CROSS_PLAY_AGENTS
)

DEFAULT_CROSS_PLAY_AGENTS = ADAPTIVE_CROSS_PLAY_AGENTS
DEFAULT_CROSS_PLAY_OPPONENT_AGENTS = ADAPTIVE_CROSS_PLAY_AGENTS

CROSS_PLAY_METADATA_FIELDNAMES = [
    "evaluation_type",
    "agent_category",
    "opponent_agent_category",
    "cross_play_matchup_type",
]

CROSS_PLAY_EVALUATION_FIELDNAMES = [
    *CHECKPOINT_EVALUATION_FIELDNAMES,
    *CROSS_PLAY_METADATA_FIELDNAMES,
]


@dataclass(frozen=True)
class CrossPlayEvaluationConfig:
    """
    Configuration for learned-agent cross-play.

    Cross-play evaluates one learned policy against another learned policy. Both
    players are model-backed and no true scripted opponent family is available,
    so adaptive players are built without expected_opponent_type and classifier
    correctness is treated as out-of-distribution diagnostic metadata.
    """

    games_per_matchup: int
    tested_agents: tuple[str, ...]
    opponent_agents: tuple[str, ...]
    eval_seed_base: int
    output_path: Path
    include_self_play: bool = False


def cross_play_agent_category(
    agent_name: str,
) -> str:
    if agent_name in ADAPTIVE_CROSS_PLAY_AGENTS:
        return "adaptive"

    if agent_name in POLICY_GENERAL_CROSS_PLAY_AGENTS:
        return "policy_general"

    raise ValueError(
        f"Unsupported cross-play agent: {agent_name}. "
        f"Supported agents: {sorted(SUPPORTED_CROSS_PLAY_AGENTS)}"
    )


def cross_play_matchup_type(
    *,
    tested_agent_name: str,
    opponent_agent_name: str,
) -> str:
    return (
        f"{cross_play_agent_category(tested_agent_name)}"
        f"_vs_{cross_play_agent_category(opponent_agent_name)}"
    )


def validate_cross_play_agent(
    agent_name: str,
) -> None:
    cross_play_agent_category(
        agent_name
    )


def should_evaluate_cross_play_matchup(
    *,
    tested_agent_name: str,
    opponent_agent_name: str,
    include_self_play: bool,
) -> bool:
    if include_self_play:
        return True

    return tested_agent_name != opponent_agent_name


def cross_play_opponent_registration_name(
    *,
    tested_agent_name: str,
    opponent_agent_name: str,
) -> str:
    if tested_agent_name == opponent_agent_name:
        return f"{opponent_agent_name}_opponent"

    return opponent_agent_name


def build_cross_play_player_loaders() -> EvaluationAgentLoaders:
    return EvaluationAgentLoaders(
        load_monte_carlo_agent=load_eval_agent,
        load_q_learning_agent=load_q_learning_eval_agent,
        load_sarsa_agent=load_sarsa_eval_agent,
        load_double_q_learning_agent=load_double_q_learning_eval_agent,
        load_monte_carlo_agents=load_adaptive_agents,
        load_q_learning_agents=load_q_learning_adaptive_agents,
        load_sarsa_agents=load_sarsa_adaptive_agents,
        load_double_q_learning_agents=load_double_q_learning_adaptive_agents,
    )


def build_cross_play_player(
    *,
    agent_name: str,
    bundle: ModelBundle,
):
    validate_cross_play_agent(
        agent_name
    )

    return build_evaluation_player(
        tested_agent_name=agent_name,
        bundle=bundle,
        loaders=build_cross_play_player_loaders(),
        expected_opponent_type=None,
        oracle_opponent_type=None,
        unsupported_context="cross-play agent",
    )


def set_cross_play_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_cross_play_seed(
    *,
    eval_seed_base: int,
    model_seed: int,
    checkpoint_episode: int,
    game_id: int,
) -> int:
    return (
        eval_seed_base
        + model_seed * 1_000_000
        + checkpoint_episode * 1_000
        + game_id
    )


def add_cross_play_metadata(
    row: dict,
    *,
    tested_agent_name: str,
    opponent_agent_name: str,
) -> dict:
    return {
        **row,
        "evaluation_type": CROSS_PLAY_EVALUATION_TYPE,
        "agent_category": cross_play_agent_category(
            tested_agent_name
        ),
        "opponent_agent_category": cross_play_agent_category(
            opponent_agent_name
        ),
        "cross_play_matchup_type": cross_play_matchup_type(
            tested_agent_name=tested_agent_name,
            opponent_agent_name=opponent_agent_name,
        ),
    }


def evaluate_single_cross_play_game(
    *,
    bundle: ModelBundle,
    tested_agent_name: str,
    opponent_agent_name: str,
    game_id: int,
    game_config: GameConfig,
    eval_seed_base: int,
) -> dict:
    if tested_agent_name == opponent_agent_name:
        registered_opponent_name = cross_play_opponent_registration_name(
            tested_agent_name=tested_agent_name,
            opponent_agent_name=opponent_agent_name,
        )
    else:
        registered_opponent_name = opponent_agent_name

    game_seed = build_cross_play_seed(
        eval_seed_base=eval_seed_base,
        model_seed=bundle.seed,
        checkpoint_episode=bundle.checkpoint_episode,
        game_id=game_id,
    )

    set_cross_play_seed(game_seed)

    config = setup_config(
        max_round=game_config.max_round,
        initial_stack=game_config.initial_stack,
        small_blind_amount=game_config.small_blind_amount,
    )

    tested_player = build_cross_play_player(
        agent_name=tested_agent_name,
        bundle=bundle,
    )

    opponent_player = build_cross_play_player(
        agent_name=opponent_agent_name,
        bundle=bundle,
    )

    config.register_player(
        name=tested_agent_name,
        algorithm=tested_player,
    )

    config.register_player(
        name=registered_opponent_name,
        algorithm=opponent_player,
    )

    result = start_poker(
        config,
        verbose=0,
    )

    hands_played = get_hands_played(
        tested_player
    )

    ended_by_bust = any(
        player_result["stack"] == 0
        for player_result in result["players"]
    )

    ended_by_round_limit = (
        not ended_by_bust
        and hands_played >= game_config.max_round
    )

    classifier_metrics = get_classifier_metrics(
        tested_player
    )

    big_blind = game_config.small_blind_amount * 2

    for player_result in result["players"]:
        if player_result["name"] == tested_agent_name:
            row = build_result_row(
                bundle=bundle,
                tested_agent_name=tested_agent_name,
                opponent_name=opponent_agent_name,
                game_id=game_id,
                final_stack=player_result["stack"],
                initial_stack=game_config.initial_stack,
                hands_played=hands_played,
                big_blind=big_blind,
                ended_by_bust=ended_by_bust,
                ended_by_round_limit=ended_by_round_limit,
                classifier_metrics=classifier_metrics,
            )

            return add_cross_play_metadata(
                row,
                tested_agent_name=tested_agent_name,
                opponent_agent_name=opponent_agent_name,
            )

    raise RuntimeError(
        "Tested cross-play player result not found in game result."
    )


def evaluate_cross_play_bundle(
    *,
    bundle: ModelBundle,
    config: CrossPlayEvaluationConfig,
) -> list[dict]:
    game_config = GameConfig()
    rows: list[dict] = []

    game_id = 0

    for tested_agent_name in config.tested_agents:
        validate_cross_play_agent(
            tested_agent_name
        )

        for opponent_agent_name in config.opponent_agents:
            validate_cross_play_agent(
                opponent_agent_name
            )

            if not should_evaluate_cross_play_matchup(
                tested_agent_name=tested_agent_name,
                opponent_agent_name=opponent_agent_name,
                include_self_play=config.include_self_play,
            ):
                continue

            for _ in range(config.games_per_matchup):
                row = evaluate_single_cross_play_game(
                    bundle=bundle,
                    tested_agent_name=tested_agent_name,
                    opponent_agent_name=opponent_agent_name,
                    game_id=game_id,
                    game_config=game_config,
                    eval_seed_base=config.eval_seed_base,
                )

                rows.append(row)
                game_id += 1

    return rows


def write_cross_play_rows(
    output_path: str | Path,
    rows: Iterable[dict],
) -> None:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CROSS_PLAY_EVALUATION_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)
