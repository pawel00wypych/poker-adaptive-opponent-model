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
from src.evaluation.runners.checkpoint_evaluator import (
    CHECKPOINT_EVALUATION_FIELDNAMES,
    ModelBundle,
    build_result_row,
    get_classifier_metrics,
    get_hands_played,
    load_adaptive_agents,
    load_eval_agent,
)
from src.evaluation.constants import (
    ADAPTIVE_MC_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    POLICY_TIGHT_AGENT,
    POLICY_GENERAL_MC_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.player_factory import (
    EvaluationAgentLoaders,
    build_evaluation_player,
    build_scripted_evaluation_player,
)


HEAD_TO_HEAD_RULE_BASED_OPPONENT = RULE_BASED_AGENT
HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT = ALWAYS_RAISE_AGENT
HEAD_TO_HEAD_ALWAYS_CALL_OPPONENT = ALWAYS_CALL_AGENT

SUPPORTED_HEAD_TO_HEAD_OPPONENTS = {
    HEAD_TO_HEAD_RULE_BASED_OPPONENT,
    HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT,
    HEAD_TO_HEAD_ALWAYS_CALL_OPPONENT,
}

DEFAULT_HEAD_TO_HEAD_OPPONENTS = (
    HEAD_TO_HEAD_RULE_BASED_OPPONENT,
    HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT,
    HEAD_TO_HEAD_ALWAYS_CALL_OPPONENT,
)

DEFAULT_HEAD_TO_HEAD_AGENTS = (
    ADAPTIVE_MC_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_TIGHT_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
)

SUPPORTED_HEAD_TO_HEAD_AGENTS = {
    ADAPTIVE_MC_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_TIGHT_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
}


@dataclass(frozen=True)
class HeadToHeadEvaluationConfig:
    """
    Configuration for direct learned-policy vs handcrafted-baseline matchups.

    This evaluator is intentionally separate from checkpoint evaluation against
    training opponents. Opponents such as rule_based and always_raise are
    out-of-distribution for the adaptive classifier, so adaptive players are
    built without an expected_opponent_type.
    """

    games_per_matchup: int
    opponents: tuple[str, ...]
    tested_agents: tuple[str, ...]
    eval_seed_base: int
    output_path: Path


def validate_head_to_head_agent(
    agent_name: str,
) -> None:
    if agent_name not in SUPPORTED_HEAD_TO_HEAD_AGENTS:
        raise ValueError(
            f"Unsupported head-to-head agent: {agent_name}. "
            f"Supported agents: {sorted(SUPPORTED_HEAD_TO_HEAD_AGENTS)}"
        )


def validate_head_to_head_opponent(
    opponent_name: str,
) -> None:
    if opponent_name not in SUPPORTED_HEAD_TO_HEAD_OPPONENTS:
        raise ValueError(
            f"Unsupported head-to-head opponent: {opponent_name}. "
            "Use rule_based, always_raise or always_call."
        )


def build_head_to_head_opponent(
    opponent_name: str,
):
    validate_head_to_head_opponent(
        opponent_name
    )

    return build_scripted_evaluation_player(
        opponent_name,
        unsupported_context="head-to-head opponent",
    )


def build_head_to_head_player_loaders() -> EvaluationAgentLoaders:
    return EvaluationAgentLoaders(
        load_monte_carlo_agent=load_eval_agent,
        load_monte_carlo_agents=load_adaptive_agents,
    )


def build_head_to_head_tested_player(
    tested_agent_name: str,
    bundle: ModelBundle,
):
    """
    Build the evaluated player for direct baseline matchups.

    Adaptive Monte Carlo is treated as an OOD evaluation here. The classifier can
    still classify into known types, but accuracy is not measured against
    rule_based/always_raise because these labels are not part of the classifier.
    """

    validate_head_to_head_agent(
        tested_agent_name
    )

    return build_evaluation_player(
        tested_agent_name=tested_agent_name,
        bundle=bundle,
        loaders=build_head_to_head_player_loaders(),
        expected_opponent_type=None,
        oracle_opponent_type=None,
        unsupported_context="head-to-head agent",
    )

def set_head_to_head_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_head_to_head_seed(
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


def evaluate_single_head_to_head_game(
    *,
    bundle: ModelBundle,
    tested_agent_name: str,
    opponent_name: str,
    game_id: int,
    game_config: GameConfig,
    eval_seed_base: int,
) -> dict:
    game_seed = build_head_to_head_seed(
        eval_seed_base=eval_seed_base,
        model_seed=bundle.seed,
        checkpoint_episode=bundle.checkpoint_episode,
        game_id=game_id,
    )

    set_head_to_head_seed(game_seed)

    config = setup_config(
        max_round=game_config.max_round,
        initial_stack=game_config.initial_stack,
        small_blind_amount=(
            game_config.small_blind_amount
        ),
    )

    tested_player = build_head_to_head_tested_player(
        tested_agent_name=tested_agent_name,
        bundle=bundle,
    )

    opponent = build_head_to_head_opponent(
        opponent_name
    )

    config.register_player(
        name=tested_agent_name,
        algorithm=tested_player,
    )

    config.register_player(
        name=opponent_name,
        algorithm=opponent,
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

    big_blind = (
        game_config.small_blind_amount * 2
    )

    for player_result in result["players"]:
        if player_result["name"] == tested_agent_name:
            return build_result_row(
                bundle=bundle,
                tested_agent_name=tested_agent_name,
                opponent_name=opponent_name,
                game_id=game_id,
                final_stack=player_result["stack"],
                initial_stack=game_config.initial_stack,
                hands_played=hands_played,
                big_blind=big_blind,
                ended_by_bust=ended_by_bust,
                ended_by_round_limit=(
                    ended_by_round_limit
                ),
                classifier_metrics=classifier_metrics,
            )

    raise RuntimeError(
        "Tested player result not found in game result."
    )


def evaluate_head_to_head_bundle(
    *,
    bundle: ModelBundle,
    config: HeadToHeadEvaluationConfig,
) -> list[dict]:
    game_config = GameConfig()
    rows: list[dict] = []

    game_id = 0

    for tested_agent_name in config.tested_agents:
        validate_head_to_head_agent(
            tested_agent_name
        )

        for opponent_name in config.opponents:
            validate_head_to_head_opponent(
                opponent_name
            )

            for _ in range(config.games_per_matchup):
                row = evaluate_single_head_to_head_game(
                    bundle=bundle,
                    tested_agent_name=(
                        tested_agent_name
                    ),
                    opponent_name=opponent_name,
                    game_id=game_id,
                    game_config=game_config,
                    eval_seed_base=config.eval_seed_base,
                )

                rows.append(row)
                game_id += 1

    return rows


def write_head_to_head_rows(
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
            fieldnames=CHECKPOINT_EVALUATION_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)
