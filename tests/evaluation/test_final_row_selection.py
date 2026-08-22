"""Validators must never choose between candidate rows.

The guidelines forbid selecting a result point using the evaluation data
itself. Aggregated validation rows are keyed by training run, agent, opponent
and training episode, so more than one candidate means the input mixes runs
that have to be validated separately - not that the better one should win.

The behavioural tests below assert ``ValueError`` rather than the concrete
exception type so that they still collect, and fail meaningfully, against a
revision where the guard does not exist yet.
"""

import pandas as pd
import pytest

from src.evaluation.validation.common import _find_row


def _row(
    agent_name="adaptive_mc",
    opponent_name="tight",
    mean_profit_bb=1.0,
    training_run="run_a",
    training_episode=1000,
    model_seed=42,
):
    return {
        "agent_name": agent_name,
        "opponent_name": opponent_name,
        "mean_profit_bb": mean_profit_bb,
        "training_run": training_run,
        "training_episode": training_episode,
        "model_seed": model_seed,
    }


def test_find_row_returns_the_single_match():
    df = pd.DataFrame(
        [
            _row(mean_profit_bb=1.5),
            _row(opponent_name="calling", mean_profit_bb=2.5),
        ]
    )

    row = _find_row(df, "adaptive_mc", "tight")

    assert row["mean_profit_bb"] == 1.5


def test_find_row_returns_none_when_no_match():
    df = pd.DataFrame([_row()])

    assert _find_row(df, "adaptive_sarsa", "tight") is None
    assert _find_row(df, "adaptive_mc", "aggressive") is None


def test_find_row_returns_none_for_empty_frame():
    df = pd.DataFrame(
        columns=[
            "agent_name",
            "opponent_name",
            "mean_profit_bb",
            "training_episode",
        ]
    )

    assert _find_row(df, "adaptive_mc", "tight") is None


def test_find_row_rejects_duplicates_instead_of_picking_the_best():
    """Previously the higher-scoring training run was silently preferred."""
    df = pd.DataFrame(
        [
            _row(training_run="run_a", mean_profit_bb=1.0),
            _row(training_run="run_b", mean_profit_bb=99.0),
        ]
    )

    with pytest.raises(ValueError) as error:
        _find_row(df, "adaptive_mc", "tight")

    message = str(error.value)
    assert "found 2" in message
    assert "training_run" in message
    assert "run_a" in message and "run_b" in message


def test_find_row_reports_duplicate_seeds():
    df = pd.DataFrame([_row(model_seed=42), _row(model_seed=123)])

    with pytest.raises(ValueError) as error:
        _find_row(df, "adaptive_mc", "tight")

    assert "model_seed" in str(error.value)


def test_find_row_filters_by_training_episode_before_uniqueness_check():
    df = pd.DataFrame(
        [
            _row(training_episode=1000, mean_profit_bb=1.0),
            _row(training_episode=2000, mean_profit_bb=99.0),
        ]
    )

    row = _find_row(df, "adaptive_mc", "tight", training_episode=1000)

    assert row["mean_profit_bb"] == 1.0


def test_find_row_rejects_duplicates_within_one_training_episode():
    df = pd.DataFrame(
        [
            _row(training_run="run_a", training_episode=1000),
            _row(training_run="run_b", training_episode=1000),
        ]
    )

    with pytest.raises(ValueError):
        _find_row(df, "adaptive_mc", "tight", training_episode=1000)


def test_ambiguity_is_reported_through_a_dedicated_exception():
    from src.evaluation.validation.common import AmbiguousValidationRowError

    assert issubclass(AmbiguousValidationRowError, ValueError)

    df = pd.DataFrame([_row(model_seed=1), _row(model_seed=2)])

    with pytest.raises(AmbiguousValidationRowError):
        _find_row(df, "adaptive_mc", "tight")


def test_best_row_selection_helper_no_longer_exists():
    """The dead idxmax-based helper must not come back."""
    import src.evaluation.validation.common as common

    assert not hasattr(common, "_best_rows_by_agent_and_opponent")
