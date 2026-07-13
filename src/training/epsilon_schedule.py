import math


def linear_epsilon(
    episode: int,
    total_episodes: int,
    epsilon_start: float,
    epsilon_min: float,
) -> float:
    """
    Linearly decrease epsilon from epsilon_start to epsilon_min.

    episode is zero-indexed:
        episode=0 -> epsilon_start
        episode=total_episodes-1 -> epsilon_min
    """
    _validate_schedule_arguments(
        episode=episode,
        total_episodes=total_episodes,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
    )

    if total_episodes == 1:
        return epsilon_min

    progress = (
        episode
        / (total_episodes - 1)
    )

    epsilon = (
        epsilon_start
        - progress
        * (
            epsilon_start
            - epsilon_min
        )
    )

    return max(
        epsilon_min,
        epsilon,
    )


def exponential_epsilon(
    episode: int,
    total_episodes: int,
    epsilon_start: float,
    epsilon_min: float,
) -> float:
    """
    Exponentially decrease epsilon so that it reaches epsilon_min
    on the final training episode.
    """
    _validate_schedule_arguments(
        episode=episode,
        total_episodes=total_episodes,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
    )

    if total_episodes == 1:
        return epsilon_min

    if epsilon_start == epsilon_min:
        return epsilon_min

    decay_rate = math.pow(
        epsilon_min / epsilon_start,
        1 / (total_episodes - 1),
    )

    epsilon = (
        epsilon_start
        * math.pow(
            decay_rate,
            episode,
        )
    )

    return max(
        epsilon_min,
        epsilon,
    )


def calculate_epsilon(
    schedule: str,
    episode: int,
    total_episodes: int,
    epsilon_start: float,
    epsilon_min: float,
) -> float:
    schedules = {
        "linear": linear_epsilon,
        "exponential": exponential_epsilon,
    }

    if schedule not in schedules:
        raise ValueError(
            f"Unsupported epsilon schedule: {schedule}"
        )

    return schedules[schedule](
        episode=episode,
        total_episodes=total_episodes,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
    )


def _validate_schedule_arguments(
    episode: int,
    total_episodes: int,
    epsilon_start: float,
    epsilon_min: float,
) -> None:
    if total_episodes <= 0:
        raise ValueError(
            "total_episodes must be greater than zero"
        )

    if not 0 <= episode < total_episodes:
        raise ValueError(
            "episode must be in range "
            "[0, total_episodes)"
        )

    if not 0 <= epsilon_min <= 1:
        raise ValueError(
            "epsilon_min must be in range [0, 1]"
        )

    if not 0 <= epsilon_start <= 1:
        raise ValueError(
            "epsilon_start must be in range [0, 1]"
        )

    if epsilon_start < epsilon_min:
        raise ValueError(
            "epsilon_start must be greater than "
            "or equal to epsilon_min"
        )