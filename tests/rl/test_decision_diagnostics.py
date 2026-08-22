"""Instrumentation for decisions that are not backed by learned values.

These tests pin two things: the counters are correct, and adding them did not
change which action is chosen. The behaviour is deliberately left as it is -
the point of this instrumentation is to measure the effect, not hide it.
"""

import random

import pytest

from src.agents.double_q_learning_agent import DoubleQLearningAgent
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SarsaAgent
from src.poker.action_mapper import ActionMapper
from src.rl.decision_diagnostics import (
    DecisionDiagnostics,
    merge_decision_diagnostics,
)

AGENT_CLASSES = (
    MonteCarloAgent,
    QLearningAgent,
    SarsaAgent,
    DoubleQLearningAgent,
)

VALID_ACTIONS = [
    {"action": "fold", "amount": 0},
    {"action": "call", "amount": 10},
    {"action": "raise", "amount": {"min": 20, "max": 200}},
]

STATE = (1, 2, 3, 0, 0, 0)


def _eval_agent(agent_class):
    agent = agent_class()
    agent.eval()
    return agent


def test_record_counts_unseen_state():
    diagnostics = DecisionDiagnostics()

    diagnostics.record(visit_counts=[0, 0, 0], action_id=1)

    assert diagnostics.decisions == 1
    assert diagnostics.unseen_state_decisions == 1
    assert diagnostics.untried_action_selections == 0


def test_record_counts_untried_action_in_a_visited_state():
    diagnostics = DecisionDiagnostics()

    diagnostics.record(visit_counts=[40, 40, 0], action_id=2)

    assert diagnostics.unseen_state_decisions == 0
    assert diagnostics.untried_action_selections == 1


def test_record_counts_a_learned_choice_as_neither():
    diagnostics = DecisionDiagnostics()

    diagnostics.record(visit_counts=[40, 40, 40], action_id=1)

    assert diagnostics.decisions == 1
    assert diagnostics.unseen_state_decisions == 0
    assert diagnostics.untried_action_selections == 0


def test_unseen_state_is_not_double_counted_as_untried_action():
    diagnostics = DecisionDiagnostics()

    diagnostics.record(visit_counts=[0, 0, 0], action_id=0)

    assert diagnostics.unseen_state_decisions == 1
    assert diagnostics.untried_action_selections == 0


def test_rates_are_zero_without_decisions():
    diagnostics = DecisionDiagnostics()

    assert diagnostics.unseen_state_decision_rate == 0.0
    assert diagnostics.untried_action_selection_rate == 0.0


def test_rates_use_the_decision_count():
    diagnostics = DecisionDiagnostics()

    diagnostics.record(visit_counts=[0, 0, 0], action_id=1)
    diagnostics.record(visit_counts=[5, 5, 5], action_id=1)
    diagnostics.record(visit_counts=[5, 5, 0], action_id=2)
    diagnostics.record(visit_counts=[5, 5, 5], action_id=1)

    assert diagnostics.unseen_state_decision_rate == pytest.approx(0.25)
    assert diagnostics.untried_action_selection_rate == pytest.approx(0.25)


def test_merge_sums_counters_across_policies():
    first = DecisionDiagnostics(
        decisions=3,
        unseen_state_decisions=1,
        untried_action_selections=1,
    )
    second = DecisionDiagnostics(
        decisions=5,
        unseen_state_decisions=2,
        untried_action_selections=0,
    )

    merged = merge_decision_diagnostics([first, second])

    assert merged.decisions == 8
    assert merged.unseen_state_decisions == 3
    assert merged.untried_action_selections == 1


@pytest.mark.parametrize("agent_class", AGENT_CLASSES)
def test_agent_counts_decisions_in_a_state_it_never_learned(agent_class):
    agent = _eval_agent(agent_class)

    agent.act(STATE, VALID_ACTIONS)

    assert agent.diagnostics.decisions == 1
    assert agent.diagnostics.unseen_state_decisions == 1


@pytest.mark.parametrize("agent_class", AGENT_CLASSES)
def test_repeated_reads_do_not_make_a_state_look_learned(agent_class):
    """Reading a Q-table inserts the state, but nothing was learned there."""
    agent = _eval_agent(agent_class)

    for _ in range(5):
        agent.act(STATE, VALID_ACTIONS)

    assert agent.diagnostics.decisions == 5
    assert agent.diagnostics.unseen_state_decisions == 5


@pytest.mark.parametrize("agent_class", AGENT_CLASSES)
def test_reset_clears_the_counters(agent_class):
    agent = _eval_agent(agent_class)
    agent.act(STATE, VALID_ACTIONS)

    agent.reset_decision_diagnostics()

    assert agent.diagnostics.decisions == 0
    assert agent.diagnostics.unseen_state_decisions == 0
    assert agent.diagnostics.untried_action_selections == 0


def test_untried_action_is_counted_for_a_partially_explored_state():
    agent = _eval_agent(MonteCarloAgent)
    agent.policy.ensure_state_exists(STATE)

    agent.q_table[STATE][ActionMapper.FOLD] = -3.0
    agent.q_table[STATE][ActionMapper.CALL] = -2.0
    agent.visit_counts[STATE][ActionMapper.FOLD] = 40
    agent.visit_counts[STATE][ActionMapper.CALL] = 40

    action_id = agent.act(STATE, VALID_ACTIONS)

    assert action_id == ActionMapper.RAISE_MIN
    assert agent.diagnostics.unseen_state_decisions == 0
    assert agent.diagnostics.untried_action_selections == 1


def test_learned_choice_is_counted_as_neither():
    agent = _eval_agent(MonteCarloAgent)
    agent.policy.ensure_state_exists(STATE)

    for action_id, value in enumerate([-3.0, -2.0, -5.0]):
        agent.q_table[STATE][action_id] = value
        agent.visit_counts[STATE][action_id] = 40

    action_id = agent.act(STATE, VALID_ACTIONS)

    assert action_id == ActionMapper.CALL
    assert agent.diagnostics.unseen_state_decisions == 0
    assert agent.diagnostics.untried_action_selections == 0


@pytest.mark.parametrize("agent_class", AGENT_CLASSES)
def test_instrumentation_does_not_change_the_chosen_actions(agent_class):
    """The counters must observe behaviour, never steer it."""
    states = [(street, 2, 3, 0, 0, 0) for street in range(4)]

    def choices_with_seed(seed):
        random.seed(seed)
        agent = _eval_agent(agent_class)
        return [agent.act(state, VALID_ACTIONS) for state in states * 10]

    baseline = choices_with_seed(1234)
    repeated = choices_with_seed(1234)

    assert baseline == repeated
