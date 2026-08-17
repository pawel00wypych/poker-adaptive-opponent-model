from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.evaluation.constants import (
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ORACLE_DOUBLE_Q_LEARNING_AGENT,
    ORACLE_MC_AGENT,
    ORACLE_Q_LEARNING_AGENT,
    ORACLE_SARSA_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
)

ALGORITHM_MONTE_CARLO = "Monte Carlo"
ALGORITHM_Q_LEARNING = "Q-learning"
ALGORITHM_SARSA = "SARSA"
ALGORITHM_DOUBLE_Q_LEARNING = "Double Q-learning"

ALGORITHM_KEY_MONTE_CARLO = "monte_carlo"
ALGORITHM_KEY_Q_LEARNING = "q_learning"
ALGORITHM_KEY_SARSA = "sarsa"
ALGORITHM_KEY_DOUBLE_Q_LEARNING = "double_q_learning"


@dataclass(frozen=True)
class AlgorithmValidationSpec:
    algorithm_key: str
    algorithm_name: str
    adaptive_agent: str
    oracle_agent: str
    general_policy_agent: str


ALGORITHM_VALIDATION_SPECS = (
    AlgorithmValidationSpec(
        algorithm_key=ALGORITHM_KEY_MONTE_CARLO,
        algorithm_name=ALGORITHM_MONTE_CARLO,
        adaptive_agent=ADAPTIVE_MC_AGENT,
        oracle_agent=ORACLE_MC_AGENT,
        general_policy_agent=POLICY_GENERAL_MC_AGENT,
    ),
    AlgorithmValidationSpec(
        algorithm_key=ALGORITHM_KEY_Q_LEARNING,
        algorithm_name=ALGORITHM_Q_LEARNING,
        adaptive_agent=ADAPTIVE_Q_LEARNING_AGENT,
        oracle_agent=ORACLE_Q_LEARNING_AGENT,
        general_policy_agent=POLICY_GENERAL_Q_LEARNING_AGENT,
    ),
    AlgorithmValidationSpec(
        algorithm_key=ALGORITHM_KEY_SARSA,
        algorithm_name=ALGORITHM_SARSA,
        adaptive_agent=ADAPTIVE_SARSA_AGENT,
        oracle_agent=ORACLE_SARSA_AGENT,
        general_policy_agent=POLICY_GENERAL_SARSA_AGENT,
    ),
    AlgorithmValidationSpec(
        algorithm_key=ALGORITHM_KEY_DOUBLE_Q_LEARNING,
        algorithm_name=ALGORITHM_DOUBLE_Q_LEARNING,
        adaptive_agent=ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
        oracle_agent=ORACLE_DOUBLE_Q_LEARNING_AGENT,
        general_policy_agent=POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    ),
)

ALGORITHM_ORDER = {
    ALGORITHM_MONTE_CARLO: 0,
    ALGORITHM_Q_LEARNING: 1,
    ALGORITHM_SARSA: 2,
    ALGORITHM_DOUBLE_Q_LEARNING: 3,
}

ADAPTIVE_AGENT_TO_ALGORITHM = {
    spec.adaptive_agent: spec.algorithm_name
    for spec in ALGORITHM_VALIDATION_SPECS
}

ORACLE_AGENT_TO_ALGORITHM = {
    spec.oracle_agent: spec.algorithm_name
    for spec in ALGORITHM_VALIDATION_SPECS
}

GENERAL_POLICY_AGENT_TO_ALGORITHM = {
    spec.general_policy_agent: spec.algorithm_name
    for spec in ALGORITHM_VALIDATION_SPECS
}

AGENT_TO_ALGORITHM = {
    **ADAPTIVE_AGENT_TO_ALGORITHM,
    **ORACLE_AGENT_TO_ALGORITHM,
    **GENERAL_POLICY_AGENT_TO_ALGORITHM,
}

ADAPTIVE_AGENTS = tuple(
    spec.adaptive_agent
    for spec in ALGORITHM_VALIDATION_SPECS
)

ORACLE_ALGORITHM_AGENTS = tuple(
    spec.oracle_agent
    for spec in ALGORITHM_VALIDATION_SPECS
)

GENERAL_POLICY_AGENTS = tuple(
    spec.general_policy_agent
    for spec in ALGORITHM_VALIDATION_SPECS
)


def algorithm_name_for_agent(agent_name: str | None) -> str | None:
    if agent_name is None:
        return None
    return AGENT_TO_ALGORITHM.get(agent_name)


def available_algorithm_specs(
    data: pd.DataFrame,
    specs: Iterable[AlgorithmValidationSpec] = ALGORITHM_VALIDATION_SPECS,
) -> tuple[AlgorithmValidationSpec, ...]:
    if data.empty or "agent_name" not in data.columns:
        return tuple()

    available_agents = set(data["agent_name"].dropna())
    return tuple(
        spec
        for spec in specs
        if (
            spec.adaptive_agent in available_agents
            or spec.oracle_agent in available_agents
            or spec.general_policy_agent in available_agents
        )
    )


def complete_algorithm_specs(
    data: pd.DataFrame,
    specs: Iterable[AlgorithmValidationSpec] = ALGORITHM_VALIDATION_SPECS,
) -> tuple[AlgorithmValidationSpec, ...]:
    if data.empty or "agent_name" not in data.columns:
        return tuple()

    available_agents = set(data["agent_name"].dropna())
    return tuple(
        spec
        for spec in specs
        if {
            spec.adaptive_agent,
            spec.oracle_agent,
            spec.general_policy_agent,
        }.issubset(available_agents)
    )

ALGORITHM_VALIDATION_SPEC_BY_KEY = {
    spec.algorithm_key: spec
    for spec in ALGORITHM_VALIDATION_SPECS
}

SUPPORTED_ALGORITHM_KEYS = tuple(ALGORITHM_VALIDATION_SPEC_BY_KEY)


def algorithm_specs_from_keys(
    algorithm_keys: Iterable[str] | None,
) -> tuple[AlgorithmValidationSpec, ...] | None:
    if algorithm_keys is None:
        return None

    unknown_keys = [
        key
        for key in algorithm_keys
        if key not in ALGORITHM_VALIDATION_SPEC_BY_KEY
    ]
    if unknown_keys:
        raise ValueError(
            f"Unsupported algorithms: {unknown_keys}. "
            f"Supported algorithms: {list(SUPPORTED_ALGORITHM_KEYS)}"
        )

    return tuple(
        ALGORITHM_VALIDATION_SPEC_BY_KEY[key]
        for key in algorithm_keys
    )
