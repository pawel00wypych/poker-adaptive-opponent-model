from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.constants import (
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    CROSS_POLICY_AGENT_TO_POLICY_TYPE,
    DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
    ORACLE_DOUBLE_Q_LEARNING_AGENT,
    ORACLE_MC_AGENT,
    ORACLE_Q_LEARNING_AGENT,
    ORACLE_SARSA_AGENT,
    Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
    RULE_BASED_AGENT,
    SARSA_POLICY_AGENT_TO_POLICY_TYPE,
)
from src.players.baselines.always_call_player import AlwaysCallPlayer
from src.players.baselines.always_raise_player import AlwaysRaisePlayer
from src.players.baselines.rule_based_player import RuleBasedPlayer
from src.players.learned.adaptive_player import AdaptivePlayer
from src.players.learned.fixed_policy_player import FixedPolicyPlayer
from src.players.learned.oracle_player import OraclePlayer

AgentLoader = Callable[[Path], Any]
AgentSetLoader = Callable[[Any], dict[str, Any]]


def _unavailable_agent_loader(path: Path) -> Any:
    raise ValueError(
        f"No model loader configured for path: {path}"
    )


def _unavailable_agent_set_loader(bundle: Any) -> dict[str, Any]:
    raise ValueError(
        "No adaptive/oracle model-set loader configured for this evaluator."
    )


@dataclass(frozen=True)
class EvaluationAgentLoaders:
    """
    Algorithm-specific loaders used by evaluation player builders.

    Evaluator modules provide these callables from their local namespace. This
    keeps monkeypatch-based tests working while moving the shared player-building
    rules into one module.
    """

    load_monte_carlo_agent: AgentLoader = _unavailable_agent_loader
    load_q_learning_agent: AgentLoader = _unavailable_agent_loader
    load_sarsa_agent: AgentLoader = _unavailable_agent_loader
    load_double_q_learning_agent: AgentLoader = _unavailable_agent_loader
    load_monte_carlo_agents: AgentSetLoader = _unavailable_agent_set_loader
    load_q_learning_agents: AgentSetLoader = _unavailable_agent_set_loader
    load_sarsa_agents: AgentSetLoader = _unavailable_agent_set_loader
    load_double_q_learning_agents: AgentSetLoader = _unavailable_agent_set_loader


def try_build_scripted_evaluation_player(agent_name: str):
    if agent_name == RULE_BASED_AGENT:
        return RuleBasedPlayer(
            player_name=RULE_BASED_AGENT,
        )

    if agent_name == ALWAYS_RAISE_AGENT:
        return AlwaysRaisePlayer(
            player_name=ALWAYS_RAISE_AGENT,
        )

    if agent_name == ALWAYS_CALL_AGENT:
        return AlwaysCallPlayer(
            player_name=ALWAYS_CALL_AGENT,
        )

    return None


def build_scripted_evaluation_player(
    agent_name: str,
    *,
    unsupported_context: str = "scripted player",
):
    player = try_build_scripted_evaluation_player(agent_name)
    if player is not None:
        return player

    raise ValueError(
        f"Unsupported {unsupported_context}: {agent_name}"
    )


def build_evaluation_player(
    *,
    tested_agent_name: str,
    bundle: Any,
    loaders: EvaluationAgentLoaders,
    expected_opponent_type: str | None,
    oracle_opponent_type: str | None,
    unsupported_context: str,
):
    scripted_player = try_build_scripted_evaluation_player(
        tested_agent_name
    )
    if scripted_player is not None:
        return scripted_player

    adaptive_loaders: dict[str, AgentSetLoader] = {
        ADAPTIVE_MC_AGENT: loaders.load_monte_carlo_agents,
        ADAPTIVE_Q_LEARNING_AGENT: loaders.load_q_learning_agents,
        ADAPTIVE_SARSA_AGENT: loaders.load_sarsa_agents,
        ADAPTIVE_DOUBLE_Q_LEARNING_AGENT: loaders.load_double_q_learning_agents,
    }

    if tested_agent_name in adaptive_loaders:
        return AdaptivePlayer(
            agents=adaptive_loaders[tested_agent_name](bundle),
            player_name=tested_agent_name,
            expected_opponent_type=expected_opponent_type,
            verbose=False,
        )

    oracle_loaders: dict[str, AgentSetLoader] = {
        ORACLE_MC_AGENT: loaders.load_monte_carlo_agents,
        ORACLE_Q_LEARNING_AGENT: loaders.load_q_learning_agents,
        ORACLE_SARSA_AGENT: loaders.load_sarsa_agents,
        ORACLE_DOUBLE_Q_LEARNING_AGENT: loaders.load_double_q_learning_agents,
    }

    if tested_agent_name in oracle_loaders:
        return OraclePlayer(
            agents=oracle_loaders[tested_agent_name](bundle),
            oracle_opponent_type=oracle_opponent_type,
            player_name=tested_agent_name,
            verbose=False,
        )

    fixed_policy_sources = (
        (
            CROSS_POLICY_AGENT_TO_POLICY_TYPE,
            bundle.agent_paths,
            loaders.load_monte_carlo_agent,
        ),
        (
            Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
            bundle.q_learning_agent_paths,
            loaders.load_q_learning_agent,
        ),
        (
            SARSA_POLICY_AGENT_TO_POLICY_TYPE,
            bundle.sarsa_agent_paths,
            loaders.load_sarsa_agent,
        ),
        (
            DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
            bundle.double_q_learning_agent_paths,
            loaders.load_double_q_learning_agent,
        ),
    )

    for policy_mapping, paths_factory, agent_loader in fixed_policy_sources:
        if tested_agent_name not in policy_mapping:
            continue

        policy_type = policy_mapping[tested_agent_name]
        agent = agent_loader(
            paths_factory()[policy_type]
        )

        return FixedPolicyPlayer(
            agent=agent,
            policy_type=policy_type,
            player_name=tested_agent_name,
            verbose=False,
        )

    raise ValueError(
        f"Unsupported {unsupported_context}: {tested_agent_name}"
    )
