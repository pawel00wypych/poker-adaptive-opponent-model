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
    load_double_q_learning_adaptive_agents,
    load_double_q_learning_eval_agent,
    load_eval_agent,
    load_q_learning_adaptive_agents,
    load_q_learning_eval_agent,
    load_sarsa_adaptive_agents,
    load_sarsa_eval_agent,
)
from src.evaluation.constants import (
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    CROSS_POLICY_AGENT_TO_POLICY_TYPE,
    ORACLE_ADAPTIVE_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    POLICY_FISH_AGENT,
    POLICY_UNKNOWN_AGENT,
    POLICY_UNKNOWN_MC_AGENT,
    POLICY_UNKNOWN_DOUBLE_Q_LEARNING_AGENT,
    POLICY_UNKNOWN_Q_LEARNING_AGENT,
    POLICY_UNKNOWN_SARSA_AGENT,
    Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
    SARSA_POLICY_AGENT_TO_POLICY_TYPE,
    DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
    RULE_BASED_AGENT,
)
from src.players.adaptive_player import AdaptivePlayer
from src.players.always_call_player import AlwaysCallPlayer
from src.players.always_raise_player import AlwaysRaisePlayer
from src.players.constants import GENERALIZATION_OPPONENTS
from src.players.fixed_policy_player import FixedPolicyPlayer
from src.players.generalization_opponents import (
    build_generalization_opponent_player,
    get_generalization_opponent_base_type,
    was_generalization_opponent_seen_during_training,
)
from src.players.oracle_adaptive_player import OracleAdaptivePlayer
from src.players.rule_based_player import RuleBasedPlayer


GENERALIZATION_EVALUATION_TYPE = "generalization"
GENERALIZATION_TRAINING_SCOPE = "base_opponents"

DEFAULT_GENERALIZATION_OPPONENTS = GENERALIZATION_OPPONENTS

DEFAULT_GENERALIZATION_AGENTS = (
    POLICY_UNKNOWN_AGENT,
    ADAPTIVE_MC_AGENT,
    ORACLE_ADAPTIVE_AGENT,
    POLICY_FISH_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    RULE_BASED_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
)

SUPPORTED_GENERALIZATION_AGENTS = set(DEFAULT_GENERALIZATION_AGENTS) | {
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    POLICY_UNKNOWN_MC_AGENT,
    POLICY_UNKNOWN_Q_LEARNING_AGENT,
    POLICY_UNKNOWN_SARSA_AGENT,
    POLICY_UNKNOWN_DOUBLE_Q_LEARNING_AGENT,
}
SUPPORTED_GENERALIZATION_OPPONENTS = set(DEFAULT_GENERALIZATION_OPPONENTS)

GENERALIZATION_METADATA_FIELDNAMES = [
    "evaluation_type",
    "trained_on",
    "seen_during_training",
    "opponent_family",
    "opponent_variant",
]

GENERALIZATION_EVALUATION_FIELDNAMES = [
    *CHECKPOINT_EVALUATION_FIELDNAMES,
    *GENERALIZATION_METADATA_FIELDNAMES,
]


@dataclass(frozen=True)
class GeneralizationEvaluationConfig:
    """
    Configuration for evaluating trained base-opponent policies on unseen
    opponent variants.

    No new specialist policies are trained for these variants. Oracle adaptive
    receives only the base opponent family, e.g. strong_calling -> calling.
    """

    games_per_matchup: int
    opponents: tuple[str, ...]
    tested_agents: tuple[str, ...]
    eval_seed_base: int
    output_path: Path


def validate_generalization_agent(
    agent_name: str,
) -> None:
    if agent_name not in SUPPORTED_GENERALIZATION_AGENTS:
        raise ValueError(
            f"Unsupported generalization agent: {agent_name}. "
            f"Supported agents: {sorted(SUPPORTED_GENERALIZATION_AGENTS)}"
        )


def validate_generalization_opponent(
    opponent_name: str,
) -> None:
    if opponent_name not in SUPPORTED_GENERALIZATION_OPPONENTS:
        raise ValueError(
            f"Unsupported generalization opponent: {opponent_name}. "
            f"Supported variants: {sorted(SUPPORTED_GENERALIZATION_OPPONENTS)}"
        )


def build_generalization_opponent(
    opponent_name: str,
    rng: random.Random | None = None,
):
    validate_generalization_opponent(
        opponent_name
    )

    return build_generalization_opponent_player(
        opponent_name=opponent_name,
        rng=rng,
    )


def build_generalization_tested_player(
    *,
    tested_agent_name: str,
    opponent_name: str,
    bundle: ModelBundle,
):
    """
    Build an evaluated player for unseen opponent-variant matchups.

    Adaptive and oracle agents use the base family label of the variant. This
    keeps the experiment aligned with the setup: trained on base opponents,
    evaluated on unseen variants, without training variant-specific specialists.
    """

    validate_generalization_agent(
        tested_agent_name
    )
    validate_generalization_opponent(
        opponent_name
    )

    opponent_family = get_generalization_opponent_base_type(
        opponent_name
    )

    if tested_agent_name == RULE_BASED_AGENT:
        return RuleBasedPlayer(
            player_name=RULE_BASED_AGENT,
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
            expected_opponent_type=opponent_family,
            verbose=False,
        )

    if tested_agent_name == ADAPTIVE_Q_LEARNING_AGENT:
        return AdaptivePlayer(
            agents=load_q_learning_adaptive_agents(bundle),
            player_name=ADAPTIVE_Q_LEARNING_AGENT,
            expected_opponent_type=opponent_family,
            verbose=False,
        )

    if tested_agent_name == ADAPTIVE_SARSA_AGENT:
        return AdaptivePlayer(
            agents=load_sarsa_adaptive_agents(bundle),
            player_name=ADAPTIVE_SARSA_AGENT,
            expected_opponent_type=opponent_family,
            verbose=False,
        )

    if tested_agent_name == ADAPTIVE_DOUBLE_Q_LEARNING_AGENT:
        return AdaptivePlayer(
            agents=load_double_q_learning_adaptive_agents(bundle),
            player_name=ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
            expected_opponent_type=opponent_family,
            verbose=False,
        )

    if tested_agent_name == ORACLE_ADAPTIVE_AGENT:
        return OracleAdaptivePlayer(
            agents=load_adaptive_agents(bundle),
            oracle_opponent_type=opponent_family,
            player_name=ORACLE_ADAPTIVE_AGENT,
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

    if tested_agent_name in Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE:
        policy_type = Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE[
            tested_agent_name
        ]

        agent = load_q_learning_eval_agent(
            bundle.q_learning_agent_paths()[policy_type]
        )

        return FixedPolicyPlayer(
            agent=agent,
            policy_type=policy_type,
            player_name=tested_agent_name,
            verbose=False,
        )

    if tested_agent_name in SARSA_POLICY_AGENT_TO_POLICY_TYPE:
        policy_type = SARSA_POLICY_AGENT_TO_POLICY_TYPE[
            tested_agent_name
        ]

        agent = load_sarsa_eval_agent(
            bundle.sarsa_agent_paths()[policy_type]
        )

        return FixedPolicyPlayer(
            agent=agent,
            policy_type=policy_type,
            player_name=tested_agent_name,
            verbose=False,
        )

    if tested_agent_name in DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE:
        policy_type = DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE[
            tested_agent_name
        ]

        agent = load_double_q_learning_eval_agent(
            bundle.double_q_learning_agent_paths()[policy_type]
        )

        return FixedPolicyPlayer(
            agent=agent,
            policy_type=policy_type,
            player_name=tested_agent_name,
            verbose=False,
        )

    raise ValueError(
        f"Unsupported generalization agent: {tested_agent_name}"
    )


def set_generalization_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_generalization_seed(
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


def add_generalization_metadata(
    row: dict,
    *,
    opponent_name: str,
) -> dict:
    opponent_family = get_generalization_opponent_base_type(
        opponent_name
    )
    seen_during_training = was_generalization_opponent_seen_during_training(
        opponent_name
    )

    return {
        **row,
        "evaluation_type": GENERALIZATION_EVALUATION_TYPE,
        "trained_on": GENERALIZATION_TRAINING_SCOPE,
        "seen_during_training": int(seen_during_training),
        "opponent_family": opponent_family,
        "opponent_variant": "" if seen_during_training else opponent_name,
    }

def evaluate_single_generalization_game(
    *,
    bundle: ModelBundle,
    tested_agent_name: str,
    opponent_name: str,
    game_id: int,
    game_config: GameConfig,
    eval_seed_base: int,
) -> dict:
    game_seed = build_generalization_seed(
        eval_seed_base=eval_seed_base,
        model_seed=bundle.seed,
        checkpoint_episode=bundle.checkpoint_episode,
        game_id=game_id,
    )

    set_generalization_seed(game_seed)

    config = setup_config(
        max_round=game_config.max_round,
        initial_stack=game_config.initial_stack,
        small_blind_amount=(
            game_config.small_blind_amount
        ),
    )

    tested_player = build_generalization_tested_player(
        tested_agent_name=tested_agent_name,
        opponent_name=opponent_name,
        bundle=bundle,
    )

    opponent = build_generalization_opponent(
        opponent_name=opponent_name,
        rng=random.Random(game_seed + 97),
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
            row = build_result_row(
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

            return add_generalization_metadata(
                row,
                opponent_name=opponent_name,
            )

    raise RuntimeError(
        "Tested player result not found in game result."
    )


def evaluate_generalization_bundle(
    *,
    bundle: ModelBundle,
    config: GeneralizationEvaluationConfig,
) -> list[dict]:
    game_config = GameConfig()
    rows: list[dict] = []

    game_id = 0

    for tested_agent_name in config.tested_agents:
        validate_generalization_agent(
            tested_agent_name
        )

        for opponent_name in config.opponents:
            validate_generalization_opponent(
                opponent_name
            )

            for _ in range(config.games_per_matchup):
                row = evaluate_single_generalization_game(
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


def write_generalization_rows(
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
            fieldnames=GENERALIZATION_EVALUATION_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
