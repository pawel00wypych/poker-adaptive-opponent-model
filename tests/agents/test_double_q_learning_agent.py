import numpy as np
import pytest

from src.agents.double_q_learning_agent import (
    DoubleQLearningAgent,
    UPDATE_Q1,
    UPDATE_Q2,
)
from src.poker.action_mapper import ActionMapper


def valid_actions_all():
    return [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]


def valid_actions_without_raise():
    return [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": -1, "max": -1}},
    ]


def test_double_q_learning_agent_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="alpha must be in range"):
        DoubleQLearningAgent(alpha=0.0)

    with pytest.raises(ValueError, match="gamma must be in range"):
        DoubleQLearningAgent(gamma=1.5)

    with pytest.raises(ValueError, match="epsilon must be in range"):
        DoubleQLearningAgent(epsilon=-0.1)

    with pytest.raises(ValueError, match="epsilon_min must be in range"):
        DoubleQLearningAgent(epsilon_min=-0.1)


def test_double_q_learning_agent_chooses_best_combined_legal_action():
    agent = DoubleQLearningAgent(epsilon=0.0)
    state = (0, 4, 2, 0, 1, 0)
    agent.q1_table[state] = np.array([-1.0, 2.0, 3.0])
    agent.q2_table[state] = np.array([0.0, 1.0, 4.0])

    action_id = agent.act(state, valid_actions_all())

    assert action_id == ActionMapper.RAISE_MIN


def test_double_q_learning_agent_does_not_choose_illegal_raise():
    agent = DoubleQLearningAgent(epsilon=0.0)
    state = (0, 4, 2, 0, 1, 0)
    agent.q1_table[state] = np.array([-1.0, 2.0, 3.0])
    agent.q2_table[state] = np.array([0.0, 1.0, 4.0])

    action_id = agent.act(state, valid_actions_without_raise())

    assert action_id == ActionMapper.CALL


def test_double_q_learning_terminal_transition_updates_q1_when_selected():
    agent = DoubleQLearningAgent(
        alpha=0.5,
        gamma=1.0,
        epsilon=0.0,
    )
    state = (0, 4, 0, 0, 3, 3, 0)

    updated_table = agent.learn_from_transition(
        state=state,
        action_id=ActionMapper.CALL,
        reward=10.0,
        done=True,
        update_table=UPDATE_Q1,
    )

    assert updated_table == UPDATE_Q1
    assert agent.q1_table[state][ActionMapper.CALL] == pytest.approx(5.0)
    assert agent.q2_table[state][ActionMapper.CALL] == pytest.approx(0.0)
    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(2.5)
    assert agent.q1_visit_counts[state][ActionMapper.CALL] == 1
    assert agent.visit_counts[state][ActionMapper.CALL] == 1


def test_double_q_learning_terminal_transition_updates_q2_when_selected():
    agent = DoubleQLearningAgent(
        alpha=0.5,
        gamma=1.0,
        epsilon=0.0,
    )
    state = (0, 4, 0, 0, 3, 3, 0)

    updated_table = agent.learn_from_transition(
        state=state,
        action_id=ActionMapper.CALL,
        reward=10.0,
        done=True,
        update_table=UPDATE_Q2,
    )

    assert updated_table == UPDATE_Q2
    assert agent.q1_table[state][ActionMapper.CALL] == pytest.approx(0.0)
    assert agent.q2_table[state][ActionMapper.CALL] == pytest.approx(5.0)
    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(2.5)
    assert agent.q2_visit_counts[state][ActionMapper.CALL] == 1
    assert agent.visit_counts[state][ActionMapper.CALL] == 1


def test_double_q_learning_randomly_selects_update_table(monkeypatch):
    agent = DoubleQLearningAgent(alpha=0.5)
    state = (0, 4, 0, 0, 3, 3, 0)

    monkeypatch.setattr(
        "src.agents.double_q_learning_agent.random.random",
        lambda: 0.25,
    )

    updated_table = agent.learn_from_transition(
        state=state,
        action_id=ActionMapper.CALL,
        reward=10.0,
        done=True,
    )

    assert updated_table == UPDATE_Q1


def test_double_q_learning_non_terminal_q1_update_uses_q1_to_select_and_q2_to_evaluate():
    agent = DoubleQLearningAgent(
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
    )
    state = (0, 4, 0, 0, 3, 3, 0)
    next_state = (1, 5, 0, 1, 2, 2, 0)

    agent.q1_table[next_state] = np.array([1.0, 5.0, 10.0])
    agent.q2_table[next_state] = np.array([3.0, 7.0, 2.0])

    agent.learn_from_transition(
        state=state,
        action_id=ActionMapper.FOLD,
        reward=2.0,
        next_state=next_state,
        next_legal_action_ids=(ActionMapper.FOLD, ActionMapper.CALL),
        done=False,
        update_table=UPDATE_Q1,
    )

    expected_target = 2.0 + 0.9 * 7.0
    expected_value = 0.5 * expected_target

    assert agent.q1_table[state][ActionMapper.FOLD] == pytest.approx(
        expected_value
    )


def test_double_q_learning_non_terminal_q2_update_uses_q2_to_select_and_q1_to_evaluate():
    agent = DoubleQLearningAgent(
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
    )
    state = (0, 4, 0, 0, 3, 3, 0)
    next_state = (1, 5, 0, 1, 2, 2, 0)

    agent.q1_table[next_state] = np.array([3.0, 7.0, 2.0])
    agent.q2_table[next_state] = np.array([1.0, 5.0, 10.0])

    agent.learn_from_transition(
        state=state,
        action_id=ActionMapper.FOLD,
        reward=2.0,
        next_state=next_state,
        next_legal_action_ids=(ActionMapper.FOLD, ActionMapper.CALL),
        done=False,
        update_table=UPDATE_Q2,
    )

    expected_target = 2.0 + 0.9 * 7.0
    expected_value = 0.5 * expected_target

    assert agent.q2_table[state][ActionMapper.FOLD] == pytest.approx(
        expected_value
    )


def test_double_q_learning_learns_from_remembered_episode(monkeypatch):
    agent = DoubleQLearningAgent(
        alpha=0.5,
        gamma=1.0,
        epsilon=0.0,
    )
    first_state = (0, 4, 0, 0, 3, 3, 0)
    second_state = (1, 5, 0, 1, 2, 2, 0)

    agent.q1_table[second_state] = np.array([0.0, 4.0, 9.0])
    agent.q2_table[second_state] = np.array([0.0, 2.0, 8.0])

    updates = iter([0.25, 0.25])
    monkeypatch.setattr(
        "src.agents.double_q_learning_agent.random.random",
        lambda: next(updates),
    )

    agent.remember(
        first_state,
        ActionMapper.CALL,
        valid_actions=valid_actions_without_raise(),
    )
    agent.remember(
        second_state,
        ActionMapper.RAISE_MIN,
        valid_actions=valid_actions_all(),
    )

    agent.learn_from_episode(reward=10.0)

    assert agent.episode == []
    assert agent.q1_table[first_state][ActionMapper.CALL] == pytest.approx(4.0)
    assert agent.q1_table[second_state][ActionMapper.RAISE_MIN] == pytest.approx(9.5)


def test_double_q_learning_eval_mode_does_not_store_or_learn():
    agent = DoubleQLearningAgent(alpha=0.5, epsilon=0.0)
    agent.eval()
    state = (0, 4, 0, 0, 3, 3, 0)

    agent.remember(
        state,
        ActionMapper.CALL,
        valid_actions=valid_actions_all(),
    )
    agent.learn_from_episode(reward=10.0)

    assert agent.episode == []
    assert state not in agent.q1_table
    assert state not in agent.q2_table


def test_double_q_learning_set_epsilon_respects_minimum():
    agent = DoubleQLearningAgent(
        epsilon=0.5,
        epsilon_min=0.05,
    )

    agent.set_epsilon(0.01)

    assert agent.epsilon == pytest.approx(0.05)


def test_double_q_learning_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "double_q_learning.pkl"
    state = (0, 4, 0, 0, 3, 3, 0)

    agent = DoubleQLearningAgent(
        alpha=0.25,
        gamma=0.9,
        epsilon=0.2,
        epsilon_min=0.05,
    )
    agent.q1_policy.set_q_value(
        state,
        ActionMapper.CALL,
        3.5,
    )
    agent.q2_policy.set_q_value(
        state,
        ActionMapper.CALL,
        1.5,
    )
    agent.q1_policy.increment_visit_count(
        state,
        ActionMapper.CALL,
    )
    agent.q2_policy.increment_visit_count(
        state,
        ActionMapper.CALL,
    )

    agent.save(
        str(path),
        metadata={"seed": 42},
    )

    loaded = DoubleQLearningAgent.load(str(path))

    assert loaded.training is False
    assert loaded.alpha == pytest.approx(0.25)
    assert loaded.gamma == pytest.approx(0.9)
    assert loaded.epsilon == pytest.approx(0.2)
    assert loaded.epsilon_min == pytest.approx(0.05)
    assert loaded.q1_table[state][ActionMapper.CALL] == pytest.approx(3.5)
    assert loaded.q2_table[state][ActionMapper.CALL] == pytest.approx(1.5)
    assert loaded.q_table[state][ActionMapper.CALL] == pytest.approx(2.5)
    assert loaded.visit_counts[state][ActionMapper.CALL] == 2
    assert DoubleQLearningAgent.load_metadata(str(path)) == {"seed": 42}
