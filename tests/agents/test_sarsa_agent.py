import numpy as np
import pytest

from src.agents.sarsa_agent import SarsaAgent
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


def test_sarsa_agent_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="alpha must be in range"):
        SarsaAgent(alpha=0.0)

    with pytest.raises(ValueError, match="gamma must be in range"):
        SarsaAgent(gamma=1.5)

    with pytest.raises(ValueError, match="epsilon must be in range"):
        SarsaAgent(epsilon=-0.1)

    with pytest.raises(ValueError, match="epsilon_min must be in range"):
        SarsaAgent(epsilon_min=-0.1)


def test_sarsa_agent_chooses_best_legal_action():
    agent = SarsaAgent(epsilon=0.0)
    state = (0, 4, 2, 0, 1, 0)
    agent.q_table[state] = np.array([-1.0, 2.0, 10.0])

    action_id = agent.act(state, valid_actions_all())

    assert action_id == ActionMapper.RAISE_MIN


def test_sarsa_agent_does_not_choose_illegal_raise():
    agent = SarsaAgent(epsilon=0.0)
    state = (0, 4, 2, 0, 1, 0)
    agent.q_table[state] = np.array([-1.0, 2.0, 10.0])

    action_id = agent.act(state, valid_actions_without_raise())

    assert action_id == ActionMapper.CALL


def test_sarsa_terminal_transition_update():
    agent = SarsaAgent(
        alpha=0.5,
        gamma=1.0,
        epsilon=0.0,
    )
    state = (0, 4, 0, 0, 3, 3, 0)

    agent.learn_from_transition(
        state=state,
        action_id=ActionMapper.CALL,
        reward=10.0,
        done=True,
    )

    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(5.0)
    assert agent.visit_counts[state][ActionMapper.CALL] == 1


def test_sarsa_non_terminal_transition_uses_actual_next_action():
    agent = SarsaAgent(
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
    )
    state = (0, 4, 0, 0, 3, 3, 0)
    next_state = (1, 5, 0, 1, 2, 2, 0)

    agent.q_table[next_state] = np.array([1.0, 5.0, 10.0])

    agent.learn_from_transition(
        state=state,
        action_id=ActionMapper.FOLD,
        reward=2.0,
        next_state=next_state,
        next_action_id=ActionMapper.CALL,
        done=False,
    )

    expected_target = 2.0 + 0.9 * 5.0
    expected_value = 0.5 * expected_target

    assert agent.q_table[state][ActionMapper.FOLD] == pytest.approx(
        expected_value
    )


def test_sarsa_does_not_use_best_next_action_when_learning():
    agent = SarsaAgent(
        alpha=0.5,
        gamma=1.0,
        epsilon=0.0,
    )
    state = (0, 4, 0, 0, 3, 3, 0)
    next_state = (1, 5, 0, 1, 2, 2, 0)

    agent.q_table[next_state] = np.array([1.0, 5.0, 10.0])

    agent.learn_from_transition(
        state=state,
        action_id=ActionMapper.CALL,
        reward=0.0,
        next_state=next_state,
        next_action_id=ActionMapper.CALL,
        done=False,
    )

    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(2.5)


def test_sarsa_learns_from_remembered_episode():
    agent = SarsaAgent(
        alpha=0.5,
        gamma=1.0,
        epsilon=0.0,
    )
    first_state = (0, 4, 0, 0, 3, 3, 0)
    second_state = (1, 5, 0, 1, 2, 2, 0)

    agent.q_table[second_state] = np.array([0.0, 4.0, 9.0])

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
    assert agent.q_table[first_state][ActionMapper.CALL] == pytest.approx(4.5)
    assert agent.q_table[second_state][ActionMapper.RAISE_MIN] == pytest.approx(9.5)


def test_sarsa_eval_mode_does_not_store_or_learn():
    agent = SarsaAgent(alpha=0.5, epsilon=0.0)
    agent.eval()
    state = (0, 4, 0, 0, 3, 3, 0)

    agent.remember(
        state,
        ActionMapper.CALL,
        valid_actions=valid_actions_all(),
    )
    agent.learn_from_episode(reward=10.0)

    assert agent.episode == []
    assert state not in agent.q_table


def test_sarsa_set_epsilon_respects_minimum():
    agent = SarsaAgent(
        epsilon=0.5,
        epsilon_min=0.05,
    )

    agent.set_epsilon(0.01)

    assert agent.epsilon == pytest.approx(0.05)


def test_sarsa_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "sarsa.pkl"
    state = (0, 4, 0, 0, 3, 3, 0)

    agent = SarsaAgent(
        alpha=0.25,
        gamma=0.9,
        epsilon=0.2,
        epsilon_min=0.05,
    )
    agent.policy.set_q_value(
        state,
        ActionMapper.CALL,
        3.5,
    )
    agent.policy.increment_visit_count(
        state,
        ActionMapper.CALL,
    )

    agent.save(
        str(path),
        metadata={"seed": 42},
    )

    loaded = SarsaAgent.load(str(path))

    assert loaded.training is False
    assert loaded.alpha == pytest.approx(0.25)
    assert loaded.gamma == pytest.approx(0.9)
    assert loaded.epsilon == pytest.approx(0.2)
    assert loaded.epsilon_min == pytest.approx(0.05)
    assert loaded.q_table[state][ActionMapper.CALL] == pytest.approx(3.5)
    assert loaded.visit_counts[state][ActionMapper.CALL] == 1
    assert SarsaAgent.load_metadata(str(path)) == {"seed": 42}
