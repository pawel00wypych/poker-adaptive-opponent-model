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


def test_peek_q_values_does_not_insert_the_state():
    policy = TabularPolicy()
    state = (1, 2, 3)

    values = policy.peek_q_values(state)

    assert list(values) == [0.0, 0.0, 0.0]
    assert len(policy.q_table) == 0
    assert len(policy.visit_counts) == 0


def test_peek_visit_counts_does_not_insert_the_state():
    policy = TabularPolicy()
    state = (1, 2, 3)

    assert policy.peek_visit_counts(state) == [0, 0, 0]
    assert len(policy.visit_counts) == 0


def test_has_state_does_not_insert_the_state():
    policy = TabularPolicy()
    state = (1, 2, 3)

    assert policy.has_state(state) is False
    assert len(policy.q_table) == 0

    policy.ensure_state_exists(state)

    assert policy.has_state(state) is True


def test_peek_returns_stored_values_for_a_known_state():
    policy = TabularPolicy()
    state = (1, 2, 3)
    policy.set_q_value(state, 1, 4.5)
    policy.increment_visit_count(state, 1)

    assert policy.peek_q_values(state)[1] == 4.5
    assert policy.peek_visit_counts(state)[1] == 1


def test_peeked_visit_counts_are_a_copy():
    policy = TabularPolicy()
    state = (1, 2, 3)
    policy.increment_visit_count(state, 0)

    peeked = policy.peek_visit_counts(state)
    peeked[0] = 999

    assert policy.get_visit_count(state, 0) == 1


def test_get_q_values_still_inserts_the_state():
    """Documents the remaining read-time mutation measured by the counters."""
    policy = TabularPolicy()

    policy.get_q_values((9, 9, 9))

    assert len(policy.q_table) == 1
