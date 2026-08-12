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

def test_agent_rejects_unknown_alpha_mode():
    with pytest.raises(
        ValueError,
        match="Unsupported alpha_mode",
    ):
        MonteCarloAgent(
            alpha_mode="linear_decay",
        )


def test_constant_alpha_updates_visit_counts():
    agent = MonteCarloAgent(
        alpha=0.1,
        epsilon=0.0,
        alpha_mode="constant",
    )

    state = (0, 4, 0, 0, 3, 3, 0)

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.learn_from_episode(
        reward=10.0,
    )

    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(
        1.0
    )
    assert agent.visit_counts[state][ActionMapper.CALL] == 1


def test_visit_count_alpha_uses_inverse_visit_count():
    agent = MonteCarloAgent(
        alpha=0.1,
        epsilon=0.0,
        alpha_mode="visit_count",
    )

    state = (0, 4, 0, 0, 3, 3, 0)

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.learn_from_episode(
        reward=10.0,
    )

    assert agent.visit_counts[state][ActionMapper.CALL] == 1
    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(
        10.0
    )

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.learn_from_episode(
        reward=0.0,
    )

    assert agent.visit_counts[state][ActionMapper.CALL] == 2
    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(
        5.0
    )

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.learn_from_episode(
        reward=5.0,
    )

    assert agent.visit_counts[state][ActionMapper.CALL] == 3
    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(
        5.0
    )


def test_sqrt_visit_alpha_uses_inverse_square_root_visit_count():
    agent = MonteCarloAgent(
        alpha=0.1,
        epsilon=0.0,
        alpha_mode="sqrt_visit",
    )

    state = (0, 4, 0, 0, 3, 3, 0)

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.learn_from_episode(
        reward=10.0,
    )

    assert agent.visit_counts[state][ActionMapper.CALL] == 1
    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(
        10.0
    )

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.learn_from_episode(
        reward=0.0,
    )

    expected = 10.0 + (1 / np.sqrt(2)) * (0.0 - 10.0)

    assert agent.visit_counts[state][ActionMapper.CALL] == 2
    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(
        expected
    )


def test_visit_counts_are_tracked_per_state_action_pair():
    agent = MonteCarloAgent(
        alpha=0.1,
        epsilon=0.0,
        alpha_mode="visit_count",
    )

    state = (0, 4, 0, 0, 3, 3, 0)

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.learn_from_episode(
        reward=10.0,
    )

    agent.remember(
        state,
        ActionMapper.RAISE_MIN,
    )
    agent.learn_from_episode(
        reward=4.0,
    )

    assert agent.visit_counts[state][ActionMapper.CALL] == 1
    assert agent.visit_counts[state][ActionMapper.RAISE_MIN] == 1
    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(
        10.0
    )
    assert agent.q_table[state][ActionMapper.RAISE_MIN] == pytest.approx(
        4.0
    )


def test_first_visit_updates_visit_count_only_once_for_repeated_pair():
    agent = MonteCarloAgent(
        alpha=0.1,
        epsilon=0.0,
        alpha_mode="visit_count",
    )

    state = (0, 4, 0, 0, 3, 3, 0)

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.learn_from_episode(
        reward=10.0,
    )

    assert agent.visit_counts[state][ActionMapper.CALL] == 1
    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(
        10.0
    )


def test_agent_save_and_load_preserves_alpha_mode_and_visit_counts(tmp_path):
    path = tmp_path / "agent.pkl"

    agent = MonteCarloAgent(
        alpha=0.1,
        epsilon=0.0,
        epsilon_min=0.05,
        alpha_mode="sqrt_visit",
    )

    state = (0, 4, 0, 0, 3, 3, 0)

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.learn_from_episode(
        reward=10.0,
    )

    agent.save(
        str(path),
        metadata={"test": True},
    )

    loaded = MonteCarloAgent.load(
        str(path)
    )

    assert loaded.training is False
    assert loaded.alpha_mode == "sqrt_visit"
    assert loaded.visit_counts[state][ActionMapper.CALL] == 1
    assert loaded.q_table[state][ActionMapper.CALL] == pytest.approx(
        10.0
    )
    assert MonteCarloAgent.load_metadata(str(path)) == {"test": True}


def test_terminal_return_is_applied_to_all_first_visited_pairs():
    agent = MonteCarloAgent(
        alpha=1.0,
        epsilon=0.0,
        alpha_mode="constant",
    )

    first_state = (0, 4, 0, 0, 3, 3, 0)
    second_state = (1, 6, 2, 1, 2, 1, 2)

    agent.remember(
        first_state,
        ActionMapper.CALL,
    )
    agent.remember(
        second_state,
        ActionMapper.RAISE_MIN,
    )
    agent.remember(
        first_state,
        ActionMapper.CALL,
    )

    agent.learn_from_episode(
        reward=3.0,
    )

    assert agent.q_table[first_state][ActionMapper.CALL] == pytest.approx(
        3.0
    )
    assert agent.q_table[second_state][ActionMapper.RAISE_MIN] == pytest.approx(
        3.0
    )
    assert agent.q_table[first_state][ActionMapper.FOLD] == pytest.approx(
        0.0
    )
    assert agent.q_table[first_state][ActionMapper.RAISE_MIN] == pytest.approx(
        0.0
    )
    assert agent.q_table[second_state][ActionMapper.FOLD] == pytest.approx(
        0.0
    )
    assert agent.q_table[second_state][ActionMapper.CALL] == pytest.approx(
        0.0
    )
    assert agent.visit_counts[first_state][ActionMapper.CALL] == 1
    assert agent.visit_counts[second_state][ActionMapper.RAISE_MIN] == 1
    assert agent.episode == []


def test_first_visit_updates_same_state_with_different_actions_separately():
    agent = MonteCarloAgent(
        alpha=1.0,
        epsilon=0.0,
        alpha_mode="constant",
    )

    state = (2, 5, 1, 2, 1, 0, 6)

    agent.remember(
        state,
        ActionMapper.CALL,
    )
    agent.remember(
        state,
        ActionMapper.RAISE_MIN,
    )
    agent.remember(
        state,
        ActionMapper.CALL,
    )

    agent.learn_from_episode(
        reward=-2.0,
    )

    assert agent.q_table[state][ActionMapper.CALL] == pytest.approx(
        -2.0
    )
    assert agent.q_table[state][ActionMapper.RAISE_MIN] == pytest.approx(
        -2.0
    )
    assert agent.q_table[state][ActionMapper.FOLD] == pytest.approx(
        0.0
    )
    assert agent.visit_counts[state][ActionMapper.CALL] == 1
    assert agent.visit_counts[state][ActionMapper.RAISE_MIN] == 1
    assert agent.visit_counts[state][ActionMapper.FOLD] == 0
    assert agent.episode == []
