import numpy as np
import pytest

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.poker.action_mapper import ActionMapper


def valid_actions_all():
    return [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]


def test_agent_creates_state_when_acting():
    agent = MonteCarloAgent(epsilon=0.0)
    state = (0, 4, 2, 0, 1, 0)

    action_id = agent.act(state, valid_actions_all())

    assert state in agent.q_table
    assert action_id in {
        ActionMapper.FOLD,
        ActionMapper.CALL,
        ActionMapper.RAISE_MIN,
    }


def test_agent_learns_positive_terminal_reward():
    agent = MonteCarloAgent(alpha=0.5, epsilon=0.0)
    state = (0, 4, 2, 0, 1, 0)

    agent.remember(state, ActionMapper.CALL)
    agent.learn_from_episode(reward=10.0)

    assert agent.q_table[state][ActionMapper.CALL] == 5.0


def test_agent_learns_negative_terminal_reward():
    agent = MonteCarloAgent(alpha=0.5, epsilon=0.0)
    state = (0, 0, 2, 0, 1, 0)

    agent.remember(state, ActionMapper.RAISE_MIN)
    agent.learn_from_episode(reward=-10.0)

    assert agent.q_table[state][ActionMapper.RAISE_MIN] == -5.0


def test_first_visit_updates_repeated_state_action_only_once():
    agent = MonteCarloAgent(alpha=0.5, epsilon=0.0)
    state = (0, 2, 2, 0, 1, 0)

    agent.remember(state, ActionMapper.CALL)
    agent.remember(state, ActionMapper.CALL)

    agent.learn_from_episode(reward=10.0)

    assert agent.q_table[state][ActionMapper.CALL] == 5.0


def test_eval_mode_does_not_store_or_learn():
    agent = MonteCarloAgent(alpha=0.5, epsilon=0.0)
    agent.eval()

    state = (0, 4, 2, 0, 1, 0)

    agent.remember(state, ActionMapper.CALL)
    agent.learn_from_episode(reward=10.0)

    assert agent.episode == []
    assert state not in agent.q_table


def test_agent_chooses_best_legal_action():
    agent = MonteCarloAgent(epsilon=0.0)
    state = (0, 4, 2, 0, 1, 0)

    agent.q_table[state] = np.array([-1.0, 2.0, 10.0])

    action_id = agent.act(state, valid_actions_all())

    assert action_id == ActionMapper.RAISE_MIN


def test_agent_does_not_choose_illegal_raise():
    agent = MonteCarloAgent(epsilon=0.0)
    state = (0, 4, 2, 0, 1, 0)

    agent.q_table[state] = np.array([-1.0, 2.0, 10.0])

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": -1, "max": -1}},
    ]

    action_id = agent.act(state, valid_actions)

    assert action_id == ActionMapper.CALL


def test_learning_does_not_change_epsilon():
    agent = MonteCarloAgent(
        epsilon=0.8,
        epsilon_min=0.1,
    )

    agent.learn_from_episode(
        reward=0.0
    )

    assert agent.epsilon == pytest.approx(
        0.8
    )


def test_agent_set_epsilon():
    agent = MonteCarloAgent(
        epsilon=0.5,
        epsilon_min=0.05,
    )

    agent.set_epsilon(
        0.25
    )

    assert agent.epsilon == pytest.approx(
        0.25
    )


def test_agent_does_not_set_epsilon_below_minimum():
    agent = MonteCarloAgent(
        epsilon=0.5,
        epsilon_min=0.05,
    )

    agent.set_epsilon(
        0.01
    )

    assert agent.epsilon == pytest.approx(
        0.05
    )


def test_learning_does_not_decay_epsilon():
    agent = MonteCarloAgent(
        epsilon=0.5,
        epsilon_min=0.05,
    )

    agent.learn_from_episode(
        reward=1.0
    )

    assert agent.epsilon == pytest.approx(
        0.5
    )