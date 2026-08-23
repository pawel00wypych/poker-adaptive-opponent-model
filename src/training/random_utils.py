"""Process-wide seeding for reproducible training runs.

``PYTHONHASHSEED`` is deliberately **not** set here. CPython reads that variable
only while the interpreter is starting, so assigning it from inside a running
process has no effect at all - it only creates false confidence. A launcher has
to put it into the environment of a **child** process before starting it;
``run_monte_carlo_suite.py`` does exactly that.

Measured, nothing in this project depends on hash ordering anyway: Q-table
dictionaries iterate in insertion order (guaranteed since Python 3.7), action
tie-breaking is built from a list rather than a set, and the only sets in the
codebase are used as ``len(set(x)) != len(x)`` duplicate checks. The variable is
therefore relevant only as a guard against future set-of-strings iteration.

The numpy global generator is not seeded either, because nothing draws from it -
verified by wrapping all 46 ``numpy.random`` entry points during a full game and
observing zero calls. The only numpy randomness in ``src/`` is
``np.random.SeedSequence``, which is a local object rather than global state.

For the randomness that games actually consume see ``src/rl/rng.py``: the
engine, the tested agent and the opponent each get their own stream.
"""

import random


def set_global_seed(
    seed: int,
) -> None:
    if seed < 0:
        raise ValueError(
            "seed must be non-negative"
        )

    random.seed(
        seed
    )
