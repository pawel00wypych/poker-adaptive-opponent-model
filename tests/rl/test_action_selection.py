import random

import numpy as np
import pytest

from src.poker.action_mapper import ActionMapper
from src.rl.action_selection import (
    get_legal_action_ids,
    select_best_legal_action,
    select_epsilon_greedy_action,
)


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


def test_get_legal_action_ids_delegates_to_action_mapper():
    assert get_legal_action_ids(valid_actions_all()) == [
        ActionMapper.FOLD,
        ActionMapper.CALL,
        ActionMapper.RAISE_MIN,
    ]
    assert get_legal_action_ids(valid_actions_without_raise()) == [
        ActionMapper.FOLD,
        ActionMapper.CALL,
    ]


def test_select_best_legal_action_ignores_illegal_high_q_value():
    q_values = np.array([-1.0, 2.0, 10.0])

    action_id = select_best_legal_action(
        q_values=q_values,
        legal_action_ids=[
            ActionMapper.FOLD,
            ActionMapper.CALL,
        ],
    )

    assert action_id == ActionMapper.CALL


def test_select_best_legal_action_breaks_ties_randomly(monkeypatch):
    monkeypatch.setattr(
        random,
        "choice",
        lambda values: values[-1],
    )

    q_values = np.array([1.0, 3.0, 3.0])

    action_id = select_best_legal_action(
        q_values=q_values,
        legal_action_ids=[
            ActionMapper.FOLD,
            ActionMapper.CALL,
            ActionMapper.RAISE_MIN,
        ],
    )

    assert action_id == ActionMapper.RAISE_MIN


def test_select_best_legal_action_rejects_empty_legal_actions():
    with pytest.raises(
        ValueError,
        match="legal_action_ids must not be empty",
    ):
        select_best_legal_action(
            q_values=np.array([0.0, 0.0, 0.0]),
            legal_action_ids=[],
        )


def test_select_epsilon_greedy_uses_best_action_in_eval_mode(monkeypatch):
    monkeypatch.setattr(
        random,
        "random",
        lambda: 0.0,
    )
    q_values = np.array([-1.0, 2.0, 10.0])

    action_id = select_epsilon_greedy_action(
        q_values=q_values,
        valid_actions=valid_actions_all(),
        epsilon=1.0,
        training=False,
    )

    assert action_id == ActionMapper.RAISE_MIN


def test_select_epsilon_greedy_explores_only_legal_actions(monkeypatch):
    monkeypatch.setattr(
        random,
        "random",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        random,
        "choice",
        lambda values: values[-1],
    )

    q_values = np.array([-1.0, 2.0, 10.0])

    action_id = select_epsilon_greedy_action(
        q_values=q_values,
        valid_actions=valid_actions_without_raise(),
        epsilon=1.0,
        training=True,
    )

    assert action_id == ActionMapper.CALL


def test_select_epsilon_greedy_rejects_invalid_epsilon():
    with pytest.raises(
        ValueError,
        match="epsilon must be in range",
    ):
        select_epsilon_greedy_action(
            q_values=np.array([0.0, 0.0, 0.0]),
            valid_actions=valid_actions_all(),
            epsilon=1.5,
            training=True,
        )
