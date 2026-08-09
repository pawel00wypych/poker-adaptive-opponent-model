import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PyPokerEngine.pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.config import GameConfig
from src.evaluation.checkpoint_evaluator import (
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
    CROSS_POLICY_AGENT_TO_POLICY_TYPE,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    POLICY_TIGHT_AGENT,
    POLICY_UNKNOWN_AGENT,
    RULE_BASED_AGENT,
    SINGLE_POLICY_MC_AGENT,
)
from src.players.adaptive_player import AdaptivePlayer
from src.players.always_call_player import AlwaysCallPlayer
from src.players.always_raise_player import AlwaysRaisePlayer
from src.players.fixed_policy_player import FixedPolicyPlayer
from src.players.rule_based_player import RuleBasedPlayer
from src.players.single_policy_player import SinglePolicyPlayer


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
    POLICY_UNKNOWN_AGENT,
    ADAPTIVE_MC_AGENT,
    POLICY_TIGHT_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
)

SUPPORTED_HEAD_TO_HEAD_AGENTS = {
    SINGLE_POLICY_MC_AGENT,
    ADAPTIVE_MC_AGENT,
    POLICY_UNKNOWN_AGENT,
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

    if opponent_name == RULE_BASED_AGENT:
        return RuleBasedPlayer(
            player_name=RULE_BASED_AGENT,
        )

    if opponent_name == ALWAYS_RAISE_AGENT:
        return AlwaysRaisePlayer(
            player_name=ALWAYS_RAISE_AGENT,
        )

    if opponent_name == ALWAYS_CALL_AGENT:
        return AlwaysCallPlayer(
            player_name=ALWAYS_CALL_AGENT,
        )

    raise ValueError(
        f"Unsupported head-to-head opponent: {opponent_name}"
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

    if tested_agent_name == SINGLE_POLICY_MC_AGENT:
        agent = load_eval_agent(
            bundle.unknown_model_path
        )

        return SinglePolicyPlayer(
            agent=agent,
            player_name=SINGLE_POLICY_MC_AGENT,
        )

    if tested_agent_name == ALWAYS_RAISE_AGENT:
        return AlwaysRaisePlayer(
            player_name=ALWAYS_RAISE_AGENT,
        )

    if tested_agent_name == ALWAYS_CALL_AGENT:
        return AlwaysCallPlayer(
            player_name=ALWAYS_CALL_AGENT,
        )

    if tested_agent_name == ADAPTIVE_MC_AGENT:
        return AdaptivePlayer(
            agents=load_adaptive_agents(bundle),
            player_name=ADAPTIVE_MC_AGENT,
            expected_opponent_type=None,
            verbose=False,
        )

    if tested_agent_name in CROSS_POLICY_AGENT_TO_POLICY_TYPE:
        policy_type = CROSS_POLICY_AGENT_TO_POLICY_TYPE[
            tested_agent_name
        ]

        agent = load_eval_agent(
            bundle.agent_paths()[policy_type]
        )

        return FixedPolicyPlayer(
            agent=agent,
            policy_type=policy_type,
            player_name=tested_agent_name,
            verbose=False,
        )

    raise ValueError(
        f"Unsupported head-to-head agent: {tested_agent_name}"
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
