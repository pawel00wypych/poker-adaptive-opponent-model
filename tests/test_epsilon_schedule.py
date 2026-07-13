import pytest

from src.training.epsilon_schedule import (
    calculate_epsilon,
    exponential_epsilon,
    linear_epsilon,
)


def test_linear_epsilon_starts_at_start_value():
    epsilon = linear_epsilon(
        episode=0,
        total_episodes=10_000,
        epsilon_start=0.5,
        epsilon_min=0.05,
    )

    assert epsilon == pytest.approx(
        0.5
    )


def test_linear_epsilon_ends_at_minimum():
    epsilon = linear_epsilon(
        episode=9_999,
        total_episodes=10_000,
        epsilon_start=0.5,
        epsilon_min=0.05,
    )

    assert epsilon == pytest.approx(
        0.05
    )


def test_linear_epsilon_is_near_middle_halfway():
    epsilon = linear_epsilon(
        episode=5_000,
        total_episodes=10_000,
        epsilon_start=0.5,
        epsilon_min=0.05,
    )

    assert epsilon == pytest.approx(
        0.274977,
        abs=0.0001,
    )


def test_exponential_epsilon_starts_at_start_value():
    epsilon = exponential_epsilon(
        episode=0,
        total_episodes=10_000,
        epsilon_start=0.5,
        epsilon_min=0.05,
    )

    assert epsilon == pytest.approx(
        0.5
    )


def test_exponential_epsilon_ends_at_minimum():
    epsilon = exponential_epsilon(
        episode=9_999,
        total_episodes=10_000,
        epsilon_start=0.5,
        epsilon_min=0.05,
    )

    assert epsilon == pytest.approx(
        0.05
    )


def test_calculate_epsilon_rejects_unknown_schedule():
    with pytest.raises(
        ValueError,
        match="Unsupported epsilon schedule",
    ):
        calculate_epsilon(
            schedule="invalid",
            episode=0,
            total_episodes=100,
            epsilon_start=0.5,
            epsilon_min=0.05,
        )


def test_schedule_rejects_invalid_episode():
    with pytest.raises(
        ValueError,
        match="episode must be in range",
    ):
        linear_epsilon(
            episode=100,
            total_episodes=100,
            epsilon_start=0.5,
            epsilon_min=0.05,
        )