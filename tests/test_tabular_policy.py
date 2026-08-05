import numpy as np
import pytest

from src.poker.action_mapper import ActionMapper
from src.rl.tabular_policy import TabularPolicy


def test_tabular_policy_creates_default_q_values_and_visit_counts():
    policy = TabularPolicy()
    state = (0, 4, 0, 0, 3, 3, 0)

    q_values = policy.get_q_values(state)

    assert state in policy.q_table
    assert state in policy.visit_counts
    assert q_values.tolist() == [0.0, 0.0, 0.0]
    assert policy.visit_counts[state] == [0, 0, 0]


def test_tabular_policy_rejects_invalid_action_count():
    with pytest.raises(
        ValueError,
        match="num_actions must be greater than zero",
    ):
        TabularPolicy(num_actions=0)


def test_tabular_policy_sets_and_gets_q_value():
    policy = TabularPolicy()
    state = (1, 5, 2, 1, 3, 0, 2)

    policy.set_q_value(
        state,
        ActionMapper.RAISE_MIN,
        3.5,
    )

    assert policy.get_q_value(
        state,
        ActionMapper.RAISE_MIN,
    ) == pytest.approx(3.5)
    assert policy.q_table[state][ActionMapper.FOLD] == pytest.approx(0.0)


def test_tabular_policy_increments_visit_counts_per_state_action_pair():
    policy = TabularPolicy()
    state = (2, 1, 0, 2, 1, 1, 3)

    first_count = policy.increment_visit_count(
        state,
        ActionMapper.CALL,
    )
    second_count = policy.increment_visit_count(
        state,
        ActionMapper.CALL,
    )

    assert first_count == 1
    assert second_count == 2
    assert policy.get_visit_count(
        state,
        ActionMapper.CALL,
    ) == 2
    assert policy.get_visit_count(
        state,
        ActionMapper.RAISE_MIN,
    ) == 0


def test_tabular_policy_converts_tables_to_plain_python_values():
    policy = TabularPolicy()
    state = (0, 4, 0, 0, 3, 3, 0)

    policy.q_table[state] = np.array([1.5, -2.0, 3.25])
    policy.visit_counts[state] = [1, 2, 3]

    plain_q_table = TabularPolicy.to_plain_q_table(
        policy.q_table
    )
    plain_visit_counts = TabularPolicy.to_plain_visit_counts(
        policy.visit_counts
    )

    assert plain_q_table == {
        state: [1.5, -2.0, 3.25]
    }
    assert plain_visit_counts == {
        state: [1, 2, 3]
    }


def test_tabular_policy_loads_plain_tables():
    policy = TabularPolicy()
    state = (0, 4, 0, 0, 3, 3, 0)

    policy.load_plain_q_table(
        {state: [1.0, 2.0, 3.0]}
    )
    policy.load_plain_visit_counts(
        {state: [4, 5, 6]}
    )

    assert policy.q_table[state].tolist() == [1.0, 2.0, 3.0]
    assert policy.visit_counts[state] == [4, 5, 6]
