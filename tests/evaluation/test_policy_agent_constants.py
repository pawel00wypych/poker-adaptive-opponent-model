"""Symmetry guarantees for the fixed specialist policy agents.

``src/evaluation/constants.py`` had no test file, which is part of why the
missing TD specialists went unnoticed: three of the four policy mappings quietly
covered only ``unknown``.
"""

import pytest

from src.evaluation.algorithm_metadata import (
    AGENT_TO_ALGORITHM,
    ALGORITHM_VALIDATION_SPECS,
)
from src.evaluation.constants import (
    AGENT_DISPLAY_NAMES,
    DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
    FIXED_SPECIALIST_AGENTS,
    MONTE_CARLO_POLICY_AGENT_TO_POLICY_TYPE,
    Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
    SARSA_POLICY_AGENT_TO_POLICY_TYPE,
    SPECIALIST_POLICY_TYPES,
    SUPPORTED_POLICY_TYPES,
    SUPPORTED_TESTED_AGENTS,
)
from src.evaluation.runners.generalization_evaluator import (
    DEFAULT_GENERALIZATION_AGENTS,
    SUPPORTED_GENERALIZATION_AGENTS,
)
from src.evaluation.runners.head_to_head_evaluator import (
    DEFAULT_HEAD_TO_HEAD_AGENTS,
    SUPPORTED_HEAD_TO_HEAD_AGENTS,
)

POLICY_MAPPINGS = {
    "monte_carlo": MONTE_CARLO_POLICY_AGENT_TO_POLICY_TYPE,
    "q_learning": Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
    "sarsa": SARSA_POLICY_AGENT_TO_POLICY_TYPE,
    "double_q_learning": DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
}


def test_every_algorithm_exposes_the_same_fixed_policy_set():
    """The bug: three mappings covered only 'unknown'.

    That made policy_tight / policy_aggressive / policy_calling Monte-Carlo-only,
    so "does a specialist generalise outside its family?" could be answered for
    one algorithm out of four.
    """
    covered = {
        name: tuple(sorted(mapping.values()))
        for name, mapping in POLICY_MAPPINGS.items()
    }

    assert len(set(covered.values())) == 1, covered
    assert set(next(iter(covered.values()))) == set(SUPPORTED_POLICY_TYPES)


@pytest.mark.parametrize("name", sorted(POLICY_MAPPINGS))
def test_each_policy_mapping_has_no_duplicate_agents(name):
    mapping = POLICY_MAPPINGS[name]

    assert len(set(mapping)) == len(mapping)


def test_policy_mappings_do_not_share_agent_names():
    """Each agent name must belong to exactly one algorithm."""
    seen = {}

    for name, mapping in POLICY_MAPPINGS.items():
        for agent in mapping:
            assert agent not in seen, f"{agent} claimed by {seen.get(agent)} and {name}"
            seen[agent] = name


def test_there_are_three_specialists_per_algorithm():
    assert len(FIXED_SPECIALIST_AGENTS) == len(POLICY_MAPPINGS) * len(
        SPECIALIST_POLICY_TYPES
    )
    assert len(set(FIXED_SPECIALIST_AGENTS)) == len(FIXED_SPECIALIST_AGENTS)


@pytest.mark.parametrize("agent", sorted(FIXED_SPECIALIST_AGENTS))
def test_every_fixed_specialist_is_a_supported_tested_agent(agent):
    assert agent in SUPPORTED_TESTED_AGENTS


@pytest.mark.parametrize("agent", sorted(FIXED_SPECIALIST_AGENTS))
def test_every_fixed_specialist_has_a_display_name(agent):
    assert agent in AGENT_DISPLAY_NAMES
    assert AGENT_DISPLAY_NAMES[agent].strip()


def test_display_names_are_unique():
    """Two agents sharing a label would be indistinguishable in a report."""
    labels = [AGENT_DISPLAY_NAMES[agent] for agent in FIXED_SPECIALIST_AGENTS]

    assert len(set(labels)) == len(labels)


@pytest.mark.parametrize("agent", sorted(FIXED_SPECIALIST_AGENTS))
def test_every_fixed_specialist_resolves_to_an_algorithm(agent):
    assert AGENT_TO_ALGORITHM.get(agent) is not None


def test_every_algorithm_spec_declares_three_specialists():
    for spec in ALGORITHM_VALIDATION_SPECS:
        assert set(spec.specialist_agent_by_policy) == set(SPECIALIST_POLICY_TYPES)
        assert len(spec.specialist_agents) == len(SPECIALIST_POLICY_TYPES)


def test_algorithm_specs_and_constants_agree():
    """The spec table and the flat tuple must not drift apart."""
    from_specs = {
        agent for spec in ALGORITHM_VALIDATION_SPECS for agent in spec.specialist_agents
    }

    assert from_specs == set(FIXED_SPECIALIST_AGENTS)


@pytest.mark.parametrize("agent", sorted(FIXED_SPECIALIST_AGENTS))
def test_fixed_specialists_are_optional_in_generalization(agent):
    """The thesis matrix keeps specialist transfer as an optional ablation."""
    assert agent not in DEFAULT_GENERALIZATION_AGENTS
    assert agent in DEFAULT_HEAD_TO_HEAD_AGENTS


@pytest.mark.parametrize("agent", sorted(FIXED_SPECIALIST_AGENTS))
def test_every_fixed_specialist_is_supported_by_the_runners(agent):
    assert agent in SUPPORTED_GENERALIZATION_AGENTS
    assert agent in SUPPORTED_HEAD_TO_HEAD_AGENTS


def test_the_monte_carlo_mapping_is_no_longer_called_cross_policy():
    """The old name read as if it spanned algorithms; it never did.

    That name is part of why the missing TD specialists were hard to see.
    """
    import src.evaluation.constants as constants

    assert not hasattr(constants, "CROSS_POLICY_AGENT_TO_POLICY_TYPE")
    assert hasattr(constants, "MONTE_CARLO_POLICY_AGENT_TO_POLICY_TYPE")


def test_player_factory_builds_every_fixed_specialist():
    """The functional check: each of the 12 must resolve to a real player.

    A name in a mapping is not enough - the factory has to reach the right
    bundle path accessor and the right loader for that algorithm.
    """
    from pathlib import Path

    from src.agents.monte_carlo_agent import MonteCarloAgent
    from src.evaluation.player_factory import (
        EvaluationAgentLoaders,
        build_evaluation_player,
    )
    from src.players.learned.fixed_policy_player import FixedPolicyPlayer

    def paths():
        return {policy: Path(f"{policy}.pkl") for policy in SUPPORTED_POLICY_TYPES}

    class StubBundle:
        agent_paths = staticmethod(paths)
        q_learning_agent_paths = staticmethod(paths)
        sarsa_agent_paths = staticmethod(paths)
        double_q_learning_agent_paths = staticmethod(paths)

    loaded = {}

    def loader_for(algorithm):
        def load(path):
            loaded[algorithm] = path
            return MonteCarloAgent()

        return load

    loaders = EvaluationAgentLoaders(
        load_monte_carlo_agent=loader_for("monte_carlo"),
        load_q_learning_agent=loader_for("q_learning"),
        load_sarsa_agent=loader_for("sarsa"),
        load_double_q_learning_agent=loader_for("double_q_learning"),
    )

    for algorithm, mapping in POLICY_MAPPINGS.items():
        for agent_name, expected_policy in mapping.items():
            loaded.clear()

            player = build_evaluation_player(
                tested_agent_name=agent_name,
                bundle=StubBundle(),
                loaders=loaders,
                expected_opponent_type=None,
                oracle_opponent_type=None,
                unsupported_context="test agent",
            )

            assert isinstance(player, FixedPolicyPlayer), agent_name
            assert player.policy_type == expected_policy, agent_name
            assert algorithm in loaded, (
                f"{agent_name} was loaded by the wrong algorithm loader: {loaded}"
            )
            assert loaded[algorithm].name == f"{expected_policy}.pkl", agent_name
