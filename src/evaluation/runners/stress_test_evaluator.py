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
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.player_factory import (
    EvaluationAgentLoaders,
    build_evaluation_player,
    build_scripted_evaluation_player,
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
from src.rl.rng import attach_rng, derive_game_streams, seed_engine_stream

STRESS_TEST_EVALUATION_TYPE = "stress_test"

STRESS_TEST_ALWAYS_CALL_OPPONENT = ALWAYS_CALL_AGENT
STRESS_TEST_ALWAYS_RAISE_OPPONENT = ALWAYS_RAISE_AGENT
STRESS_TEST_RULE_BASED_OPPONENT = RULE_BASED_AGENT

SUPPORTED_STRESS_TEST_OPPONENTS = {
    STRESS_TEST_ALWAYS_CALL_OPPONENT,
    STRESS_TEST_ALWAYS_RAISE_OPPONENT,
    STRESS_TEST_RULE_BASED_OPPONENT,
}

DEFAULT_STRESS_TEST_OPPONENTS = (
    STRESS_TEST_ALWAYS_CALL_OPPONENT,
    STRESS_TEST_ALWAYS_RAISE_OPPONENT,
    STRESS_TEST_RULE_BASED_OPPONENT,
)

DEFAULT_STRESS_TEST_AGENTS = (
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
)

SUPPORTED_STRESS_TEST_AGENTS = set(DEFAULT_STRESS_TEST_AGENTS) | {
    RULE_BASED_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
}

STRESS_TEST_METADATA_FIELDNAMES = [
    "evaluation_type",
    "stress_opponent_type",
]

STRESS_TEST_EVALUATION_FIELDNAMES = [
    *MODEL_EVALUATION_FIELDNAMES,
    *STRESS_TEST_METADATA_FIELDNAMES,
]


@dataclass(frozen=True)
class StressTestEvaluationConfig:
    """
    Configuration for stress-testing learned policies against handcrafted
    extreme/sanity opponents.

    Stress-test opponents are intentionally treated as out-of-distribution for
    adaptive classifier accuracy. Adaptive players can still classify and switch
    policies, but expected_opponent_type is not provided because always_call,
    always_raise and rule_based are not classifier target families.
    """

    games_per_matchup: int
    opponents: tuple[str, ...]
    tested_agents: tuple[str, ...]
    eval_seed_base: int
    output_path: Path


def validate_stress_test_agent(
    agent_name: str,
) -> None:
    if agent_name not in SUPPORTED_STRESS_TEST_AGENTS:
        raise ValueError(
            f"Unsupported stress-test agent: {agent_name}. "
            f"Supported agents: {sorted(SUPPORTED_STRESS_TEST_AGENTS)}"
        )


def validate_stress_test_opponent(
    opponent_name: str,
) -> None:
    if opponent_name not in SUPPORTED_STRESS_TEST_OPPONENTS:
        raise ValueError(
            f"Unsupported stress-test opponent: {opponent_name}. "
            "Use always_call, always_raise or rule_based."
        )


def build_stress_test_opponent(
    opponent_name: str,
):
    validate_stress_test_opponent(opponent_name)

    return build_scripted_evaluation_player(
        opponent_name,
        unsupported_context="stress-test opponent",
    )


def build_stress_test_player_loaders() -> EvaluationAgentLoaders:
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


def build_stress_tested_player(
    *,
    tested_agent_name: str,
    bundle: ModelBundle,
):
    validate_stress_test_agent(tested_agent_name)

    return build_evaluation_player(
        tested_agent_name=tested_agent_name,
        bundle=bundle,
        loaders=build_stress_test_player_loaders(),
        expected_opponent_type=None,
        oracle_opponent_type=None,
        unsupported_context="stress-test agent",
    )


def build_stress_test_seed(
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


def stress_test_opponent_registration_name(
    *,
    tested_agent_name: str,
    opponent_name: str,
) -> str:
    if tested_agent_name == opponent_name:
        return f"{opponent_name}_opponent"

    return opponent_name


def add_stress_test_metadata(
    row: dict,
    *,
    opponent_name: str,
) -> dict:
    return {
        **row,
        "evaluation_type": STRESS_TEST_EVALUATION_TYPE,
        "stress_opponent_type": opponent_name,
    }


def evaluate_single_stress_test_game(
    *,
    bundle: ModelBundle,
    tested_agent_name: str,
    opponent_name: str,
    game_id: int,
    matchup_game_index: int,
    game_config: GameConfig,
    eval_seed_base: int,
) -> dict:
    game_seed = build_stress_test_seed(
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
        small_blind_amount=game_config.small_blind_amount,
    )

    tested_player = build_stress_tested_player(
        tested_agent_name=tested_agent_name,
        bundle=bundle,
    )

    opponent = build_stress_test_opponent(opponent_name)

    attach_rng(tested_player, streams.agent)
    attach_rng(opponent, streams.opponent)

    registered_opponent_name = stress_test_opponent_registration_name(
        tested_agent_name=tested_agent_name,
        opponent_name=opponent_name,
    )

    config.register_player(
        name=tested_agent_name,
        algorithm=tested_player,
    )

    config.register_player(
        name=registered_opponent_name,
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
                ended_by_round_limit=ended_by_round_limit,
                classifier_metrics=classifier_metrics,
                decision_diagnostics=decision_diagnostics,
            )

            return add_stress_test_metadata(
                row,
                opponent_name=opponent_name,
            )

    raise RuntimeError("Tested player result not found in game result.")


def evaluate_stress_test_bundle(
    *,
    bundle: ModelBundle,
    config: StressTestEvaluationConfig,
) -> list[dict]:
    game_config = GameConfig()
    rows: list[dict] = []

    game_id = 0

    for tested_agent_name in config.tested_agents:
        validate_stress_test_agent(tested_agent_name)

        for opponent_name in config.opponents:
            validate_stress_test_opponent(opponent_name)

            for matchup_game_index in range(config.games_per_matchup):
                row = evaluate_single_stress_test_game(
                    bundle=bundle,
                    tested_agent_name=tested_agent_name,
                    opponent_name=opponent_name,
                    game_id=game_id,
                    matchup_game_index=matchup_game_index,
                    game_config=game_config,
                    eval_seed_base=config.eval_seed_base,
                )

                rows.append(row)
                game_id += 1

    return rows


def write_stress_test_rows(
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
            fieldnames=STRESS_TEST_EVALUATION_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)
