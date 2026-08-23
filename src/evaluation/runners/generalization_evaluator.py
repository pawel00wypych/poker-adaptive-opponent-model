import csv
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    ORACLE_DOUBLE_Q_LEARNING_AGENT,
    ORACLE_MC_AGENT,
    ORACLE_Q_LEARNING_AGENT,
    ORACLE_SARSA_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
    POLICY_TIGHT_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.player_factory import (
    EvaluationAgentLoaders,
    build_evaluation_player,
)
from src.evaluation.runners.evaluation_seed import (
    build_paired_evaluation_seed,
)
from src.evaluation.runners.model_evaluator import (
    MODEL_EVALUATION_FIELDNAMES,
    ModelBundle,
    build_result_row,
    get_classifier_metrics,
    get_decision_diagnostics,
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
from src.players.constants import GENERALIZATION_OPPONENTS
from src.players.generalization.generalization_opponents import (
    build_generalization_opponent_player,
    get_generalization_opponent_base_type,
    was_generalization_opponent_seen_during_training,
)
from src.rl.rng import attach_rng, derive_game_streams, seed_engine_stream

GENERALIZATION_EVALUATION_TYPE = "generalization"
GENERALIZATION_TRAINING_SCOPE = "base_opponents"

DEFAULT_GENERALIZATION_OPPONENTS = GENERALIZATION_OPPONENTS

DEFAULT_GENERALIZATION_AGENTS = (
    ADAPTIVE_MC_AGENT,
    ORACLE_MC_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_TIGHT_AGENT,
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
    ORACLE_Q_LEARNING_AGENT,
    ORACLE_SARSA_AGENT,
    ORACLE_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
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
    *MODEL_EVALUATION_FIELDNAMES,
    *GENERALIZATION_METADATA_FIELDNAMES,
]


@dataclass(frozen=True)
class GeneralizationEvaluationConfig:
    """
    Configuration for evaluating trained base-opponent policies on unseen
    opponent variants.

    No new specialist policies are trained for these variants. Oracle adaptive
    receives only the base opponent family, e.g. calling_extreme -> calling.
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
    validate_generalization_opponent(opponent_name)

    return build_generalization_opponent_player(
        opponent_name=opponent_name,
        rng=rng,
    )


def build_generalization_player_loaders() -> EvaluationAgentLoaders:
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

    validate_generalization_agent(tested_agent_name)
    validate_generalization_opponent(opponent_name)

    opponent_family = get_generalization_opponent_base_type(opponent_name)

    return build_evaluation_player(
        tested_agent_name=tested_agent_name,
        bundle=bundle,
        loaders=build_generalization_player_loaders(),
        expected_opponent_type=opponent_family,
        oracle_opponent_type=opponent_family,
        unsupported_context="generalization agent",
    )


def build_generalization_seed(
    *,
    eval_seed_base: int,
    model_seed: int,
    model_episode: int,
    matchup_game_index: int,
) -> int:
    return build_paired_evaluation_seed(
        eval_seed_base=eval_seed_base,
        model_seed=model_seed,
        model_episode=model_episode,
        matchup_game_index=matchup_game_index,
    )


def add_generalization_metadata(
    row: dict,
    *,
    opponent_name: str,
) -> dict:
    opponent_family = get_generalization_opponent_base_type(opponent_name)
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
    matchup_game_index: int,
    game_config: GameConfig,
    eval_seed_base: int,
) -> dict:
    game_seed = build_generalization_seed(
        eval_seed_base=eval_seed_base,
        model_seed=bundle.seed,
        model_episode=bundle.model_episode,
        matchup_game_index=matchup_game_index,
    )

    streams = derive_game_streams(game_seed)

    seed_engine_stream(streams.deck_seed)

    config = setup_config(
        max_round=game_config.max_round,
        initial_stack=game_config.initial_stack,
        small_blind_amount=(game_config.small_blind_amount),
    )

    tested_player = build_generalization_tested_player(
        tested_agent_name=tested_agent_name,
        opponent_name=opponent_name,
        bundle=bundle,
    )

    opponent = build_generalization_opponent(
        opponent_name=opponent_name,
        rng=streams.opponent,
    )

    attach_rng(tested_player, streams.agent)
    attach_rng(opponent, streams.opponent)

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

    hands_played = get_hands_played(tested_player)

    ended_by_bust = any(
        player_result["stack"] == 0 for player_result in result["players"]
    )

    ended_by_round_limit = not ended_by_bust and hands_played >= game_config.max_round

    classifier_metrics = get_classifier_metrics(tested_player)
    decision_diagnostics = get_decision_diagnostics(tested_player)

    big_blind = game_config.small_blind_amount * 2

    for player_result in result["players"]:
        if player_result["name"] == tested_agent_name:
            row = build_result_row(
                bundle=bundle,
                tested_agent_name=tested_agent_name,
                opponent_name=opponent_name,
                game_id=game_id,
                matchup_game_index=matchup_game_index,
                evaluation_seed=game_seed,
                final_stack=player_result["stack"],
                initial_stack=game_config.initial_stack,
                hands_played=hands_played,
                big_blind=big_blind,
                ended_by_bust=ended_by_bust,
                ended_by_round_limit=(ended_by_round_limit),
                classifier_metrics=classifier_metrics,
                decision_diagnostics=decision_diagnostics,
            )

            return add_generalization_metadata(
                row,
                opponent_name=opponent_name,
            )

    raise RuntimeError("Tested player result not found in game result.")


def evaluate_generalization_bundle(
    *,
    bundle: ModelBundle,
    config: GeneralizationEvaluationConfig,
) -> list[dict]:
    game_config = GameConfig()
    rows: list[dict] = []

    game_id = 0

    for tested_agent_name in config.tested_agents:
        validate_generalization_agent(tested_agent_name)

        for opponent_name in config.opponents:
            validate_generalization_opponent(opponent_name)

            for matchup_game_index in range(config.games_per_matchup):
                row = evaluate_single_generalization_game(
                    bundle=bundle,
                    tested_agent_name=(tested_agent_name),
                    opponent_name=opponent_name,
                    game_id=game_id,
                    matchup_game_index=matchup_game_index,
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
