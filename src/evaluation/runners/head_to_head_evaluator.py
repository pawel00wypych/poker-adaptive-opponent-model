import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.config import GameConfig
from src.evaluation.constants import (
    ADAPTIVE_MC_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_TIGHT_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.player_factory import (
    EvaluationAgentLoaders,
    build_evaluation_player,
    build_scripted_evaluation_player,
)
from src.evaluation.runners.evaluation_seed import (
    build_baseline_evaluation_seed,
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
    load_eval_agent,
)
from src.rl.rng import attach_rng, derive_game_streams, seed_engine_stream

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
    RULE_BASED_AGENT,
)

BASELINE_ONLY_AGENTS = (
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    RULE_BASED_AGENT,
)

BASELINE_ONLY_AGENT_SET = frozenset(BASELINE_ONLY_AGENTS)

SUPPORTED_HEAD_TO_HEAD_AGENTS = {
    ADAPTIVE_MC_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_TIGHT_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    RULE_BASED_AGENT,
}


@dataclass(frozen=True)
class HeadToHeadEvaluationConfig:
    """
    Configuration for direct learned-policy vs handcrafted-baseline matchups.

    This evaluator is intentionally separate from final-model evaluation against
    training opponents. Opponents such as rule_based and always_raise are
    out-of-distribution for the adaptive classifier, so adaptive players are
    built without an expected_opponent_type.
    """

    games_per_matchup: int
    opponents: tuple[str, ...]
    tested_agents: tuple[str, ...]
    eval_seed_base: int
    output_path: Path
    evaluation_replicates: int = 5

    def __post_init__(self) -> None:
        if self.games_per_matchup <= 0:
            raise ValueError("games_per_matchup must be greater than zero")
        if self.evaluation_replicates <= 0:
            raise ValueError("evaluation_replicates must be greater than zero")


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
    validate_head_to_head_opponent(opponent_name)

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
    bundle: ModelBundle | None,
):
    """
    Build the evaluated player for direct baseline matchups.

    Adaptive Monte Carlo is treated as an OOD evaluation here. The classifier can
    still classify into known types, but accuracy is not measured against
    rule_based/always_raise because these labels are not part of the classifier.
    """

    validate_head_to_head_agent(tested_agent_name)

    if tested_agent_name in BASELINE_ONLY_AGENT_SET:
        return build_scripted_evaluation_player(
            tested_agent_name,
            unsupported_context="head-to-head agent",
        )

    if bundle is None:
        raise ValueError(
            f"Model bundle is required for learned agent: {tested_agent_name}"
        )

    return build_evaluation_player(
        tested_agent_name=tested_agent_name,
        bundle=bundle,
        loaders=build_head_to_head_player_loaders(),
        expected_opponent_type=None,
        oracle_opponent_type=None,
        unsupported_context="head-to-head agent",
    )


def baseline_tested_agents(
    agent_names: Iterable[str],
) -> tuple[str, ...]:
    return tuple(
        agent_name
        for agent_name in agent_names
        if agent_name in BASELINE_ONLY_AGENT_SET
    )


def learned_tested_agents(
    agent_names: Iterable[str],
) -> tuple[str, ...]:
    return tuple(
        agent_name
        for agent_name in agent_names
        if agent_name not in BASELINE_ONLY_AGENT_SET
    )


def build_head_to_head_seed(
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


def evaluate_single_head_to_head_game(
    *,
    bundle: ModelBundle,
    tested_agent_name: str,
    opponent_name: str,
    game_id: int,
    matchup_game_index: int,
    game_config: GameConfig,
    eval_seed_base: int,
) -> dict:
    if tested_agent_name in BASELINE_ONLY_AGENT_SET:
        raise ValueError("Baseline-only agents must use evaluate_single_baseline_game")

    game_seed = build_head_to_head_seed(
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

    tested_player = build_head_to_head_tested_player(
        tested_agent_name=tested_agent_name,
        bundle=bundle,
    )

    opponent = build_head_to_head_opponent(opponent_name)

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

    big_blind = game_config.big_blind_amount

    for player_result in result["players"]:
        if player_result["name"] == tested_agent_name:
            return build_result_row(
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

    raise RuntimeError("Tested player result not found in game result.")


def evaluate_head_to_head_bundle(
    *,
    bundle: ModelBundle,
    config: HeadToHeadEvaluationConfig,
) -> list[dict]:
    game_config = GameConfig()
    rows: list[dict] = []

    game_id = 0

    for tested_agent_name in learned_tested_agents(config.tested_agents):
        validate_head_to_head_agent(tested_agent_name)

        for opponent_name in config.opponents:
            validate_head_to_head_opponent(opponent_name)

            for matchup_game_index in range(config.games_per_matchup):
                row = evaluate_single_head_to_head_game(
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


def evaluate_single_baseline_game(
    *,
    tested_agent_name: str,
    opponent_name: str,
    evaluation_replicate_id: int,
    game_id: int,
    matchup_game_index: int,
    game_config: GameConfig,
    eval_seed_base: int,
) -> dict:
    if tested_agent_name not in BASELINE_ONLY_AGENT_SET:
        raise ValueError(
            f"Baseline-only evaluation does not support: {tested_agent_name}"
        )

    game_seed = build_baseline_evaluation_seed(
        eval_seed_base=eval_seed_base,
        evaluation_replicate_id=evaluation_replicate_id,
        matchup_game_index=matchup_game_index,
    )
    streams = derive_game_streams(game_seed)
    seed_engine_stream(streams.deck_seed)

    config = setup_config(
        max_round=game_config.max_round,
        initial_stack=game_config.initial_stack,
        small_blind_amount=game_config.small_blind_amount,
    )
    tested_player = build_head_to_head_tested_player(
        tested_agent_name=tested_agent_name,
        bundle=None,
    )
    opponent = build_head_to_head_opponent(opponent_name)

    attach_rng(tested_player, streams.agent)
    attach_rng(opponent, streams.opponent)

    registered_opponent_name = (
        f"{opponent_name}_opponent"
        if tested_agent_name == opponent_name
        else opponent_name
    )

    config.register_player(
        name=tested_agent_name,
        algorithm=tested_player,
    )
    config.register_player(
        name=registered_opponent_name,
        algorithm=opponent,
    )

    result = start_poker(config, verbose=0)
    hands_played = get_hands_played(tested_player)
    ended_by_bust = any(
        player_result["stack"] == 0 for player_result in result["players"]
    )
    ended_by_round_limit = not ended_by_bust and hands_played >= game_config.max_round
    classifier_metrics = get_classifier_metrics(tested_player)
    decision_diagnostics = get_decision_diagnostics(tested_player)
    big_blind = game_config.big_blind_amount

    for player_result in result["players"]:
        if player_result["name"] == tested_agent_name:
            return build_result_row(
                bundle=None,
                tested_agent_name=tested_agent_name,
                opponent_name=opponent_name,
                game_id=game_id,
                matchup_game_index=matchup_game_index,
                evaluation_seed=game_seed,
                evaluation_replicate_id=evaluation_replicate_id,
                final_stack=player_result["stack"],
                initial_stack=game_config.initial_stack,
                hands_played=hands_played,
                big_blind=big_blind,
                ended_by_bust=ended_by_bust,
                ended_by_round_limit=ended_by_round_limit,
                classifier_metrics=classifier_metrics,
                decision_diagnostics=decision_diagnostics,
            )

    raise RuntimeError("Tested baseline result not found in game result.")


def evaluate_baseline_replicate(
    *,
    evaluation_replicate_id: int,
    config: HeadToHeadEvaluationConfig,
) -> list[dict]:
    game_config = GameConfig()
    rows: list[dict] = []
    tested_agents = baseline_tested_agents(config.tested_agents)
    games_per_replicate = (
        len(tested_agents) * len(config.opponents) * config.games_per_matchup
    )
    game_id = evaluation_replicate_id * games_per_replicate

    for tested_agent_name in tested_agents:
        validate_head_to_head_agent(tested_agent_name)

        for opponent_name in config.opponents:
            validate_head_to_head_opponent(opponent_name)

            for matchup_game_index in range(config.games_per_matchup):
                rows.append(
                    evaluate_single_baseline_game(
                        tested_agent_name=tested_agent_name,
                        opponent_name=opponent_name,
                        evaluation_replicate_id=evaluation_replicate_id,
                        game_id=game_id,
                        matchup_game_index=matchup_game_index,
                        game_config=game_config,
                        eval_seed_base=config.eval_seed_base,
                    )
                )
                game_id += 1

    return rows


def evaluate_baseline_replicates(
    *,
    config: HeadToHeadEvaluationConfig,
) -> list[dict]:
    rows: list[dict] = []

    for evaluation_replicate_id in range(config.evaluation_replicates):
        rows.extend(
            evaluate_baseline_replicate(
                evaluation_replicate_id=evaluation_replicate_id,
                config=config,
            )
        )

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
            fieldnames=MODEL_EVALUATION_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)
