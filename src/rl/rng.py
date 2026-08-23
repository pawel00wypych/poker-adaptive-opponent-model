"""Independent random streams for the engine, the tested agent and the opponent.

PyPokerEngine shuffles the deck with the module-level ``random`` module and
offers no seed-injection point - ``pypokerengine/engine/deck.py`` calls
``random.shuffle`` directly. The engine therefore has to own the global stream,
and everything else has to be moved off it.

This is not a cosmetic separation. The deck is reshuffled once per hand from the
same stream that agents and opponents draw from, so a single extra
``random.random()`` call inside an agent shifts every later shuffle in the game.
Measured on two agents that make identical decisions and differ only in how many
draws they take::

    extra draws/decision | identical hole cards? | first divergence
                       0 |                  True | -
                       1 |                 False | decision 4

That silently defeats the common-random-numbers design that
``build_paired_evaluation_seed`` documents, and it means the compared algorithms
train on different hands, because they consume different numbers of draws.

Substreams are derived with ``SeedSequence.spawn``, which is the same primitive
``evaluation_seed.py`` already uses, rather than ad-hoc offsets such as
``game_seed + 97``: offsets from nearby seeds are not guaranteed to be
independent, while spawned sequences are designed to be.
"""

import random

import numpy as np

DECK_STREAM = "deck"
AGENT_STREAM = "agent"
OPPONENT_STREAM = "opponent"

STREAM_NAMES = (DECK_STREAM, AGENT_STREAM, OPPONENT_STREAM)


class GameStreams:
    """One independent random stream per source of randomness in a game."""

    __slots__ = ("deck_seed", "agent", "opponent")

    def __init__(
        self,
        *,
        deck_seed: int,
        agent: random.Random,
        opponent: random.Random,
    ) -> None:
        self.deck_seed = deck_seed
        self.agent = agent
        self.opponent = opponent


def _seed_from(seed_sequence: np.random.SeedSequence) -> int:
    state = seed_sequence.generate_state(4, dtype=np.uint32)
    return int.from_bytes(state.tobytes(), "little")


def derive_game_streams(game_seed: int) -> GameStreams:
    """Split one game seed into three mutually independent streams.

    The deck stream is returned as a plain integer because it has to be pushed
    into the global ``random`` module, which is the only way to reach the
    engine's shuffle.
    """
    if isinstance(game_seed, bool) or not isinstance(game_seed, int):
        raise TypeError("game_seed must be an integer")

    if game_seed < 0:
        raise ValueError("game_seed must be non-negative")

    return _streams_from(np.random.SeedSequence(game_seed))


def derive_episode_streams(training_seed: int, episode_index: int) -> GameStreams:
    """Split one training episode into three mutually independent streams.

    Keying on ``(training_seed, episode_index)`` rather than on position in a
    single continuous stream has two consequences that matter for the thesis:

    * every algorithm is dealt the same hands for a given seed and episode,
      even though they consume different numbers of draws - Double Q-learning
      in particular flips a coin on every update;
    * an episode becomes independently reproducible instead of depending on the
      entire preceding run.
    """
    for name, value in (
        ("training_seed", training_seed),
        ("episode_index", episode_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    return _streams_from(
        np.random.SeedSequence([training_seed, episode_index])
    )


def _streams_from(root: np.random.SeedSequence) -> GameStreams:
    deck_sequence, agent_sequence, opponent_sequence = root.spawn(3)

    return GameStreams(
        deck_seed=_seed_from(deck_sequence),
        agent=random.Random(_seed_from(agent_sequence)),
        opponent=random.Random(_seed_from(opponent_sequence)),
    )


def seed_engine_stream(deck_seed: int) -> None:
    """Point the global ``random`` module at the deck stream.

    The engine can only be reached this way, so after this call nothing else
    may draw from the global module without perturbing the shuffle.
    """
    random.seed(deck_seed)


def attach_rng(player, rng: random.Random) -> None:
    """Point a player, and any agents it owns, at one private random stream.

    Players hold their agents either as ``agent`` or as an ``agents`` mapping,
    so both shapes are handled. Scripted players that never draw are still given
    the attribute, which keeps the wiring uniform and makes an unattached player
    easy to spot.
    """
    player.rng = rng

    agent = getattr(player, "agent", None)
    if agent is not None:
        agent.rng = rng

    for owned_agent in getattr(player, "agents", {}).values():
        owned_agent.rng = rng
