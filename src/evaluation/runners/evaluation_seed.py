from numbers import Integral

import numpy as np


def build_paired_evaluation_seed(
    *,
    eval_seed_base: int,
    model_seed: int,
    checkpoint_episode: int,
    matchup_game_index: int,
) -> int:
    """Build a reproducible seed shared by corresponding matchup games.

    Agent and opponent names are deliberately not part of the seed. This gives
    every matchup for the same model seed and checkpoint the same sequence of
    evaluation seeds, enabling common-random-number comparisons.
    """
    components = {
        "eval_seed_base": eval_seed_base,
        "model_seed": model_seed,
        "checkpoint_episode": checkpoint_episode,
        "matchup_game_index": matchup_game_index,
    }

    for name, value in components.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")

        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    seed_sequence = np.random.SeedSequence(
        [int(value) for value in components.values()]
    )

    return int(
        seed_sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )
