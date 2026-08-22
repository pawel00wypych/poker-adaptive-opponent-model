from pathlib import Path
from typing import Iterable


def build_checkpoint_episodes(
    total_episodes: int,
    configured_checkpoints: Iterable[int],
    checkpoint_interval: int | None = None,
) -> set[int]:
    if total_episodes <= 0:
        raise ValueError(
            "total_episodes must be greater than zero"
        )

    if checkpoint_interval is not None:
        if checkpoint_interval <= 0:
            raise ValueError(
                "checkpoint_interval must be greater than zero"
            )

        checkpoints = set(
            range(
                checkpoint_interval,
                total_episodes + 1,
                checkpoint_interval,
            )
        )
    else:
        checkpoints = {
            checkpoint
            for checkpoint in configured_checkpoints
            if 0 < checkpoint <= total_episodes
        }

    checkpoints.add(total_episodes)

    return checkpoints


def build_checkpoint_path(
    checkpoint_directory: str,
    model_name: str,
    completed_episodes: int,
    seed: int,
) -> str:
    directory = Path(checkpoint_directory)

    filename = (
        f"{model_name}"
        f"_episodes_{completed_episodes}"
        f"_seed_{seed}.pkl"
    )

    return str(directory / filename)
