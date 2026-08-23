"""Guards for the separation of the engine, agent and opponent random streams.

Every test here that plays real hands compares cards **per hand**, never per
decision. Different agents take different numbers of decisions inside a hand, so
comparing by decision index conflates a genuine card difference with a
legitimate difference in how the hand was played.
"""

import random

import pytest
from pypokerengine.api.game import setup_config, start_poker

from src.agents.double_q_learning_agent import DoubleQLearningAgent
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SarsaAgent
from src.players.base.player_template import PlayerTemplate
from src.players.learned.general_policy_player import GeneralPolicyPlayer
from src.players.opponents.calling_player import CallingPlayer
from src.players.opponents.factory import build_opponent
from src.poker.constants import TRAINING_OPPONENT_TYPES
from src.rl.rng import (
    attach_rng,
    derive_episode_streams,
    derive_game_streams,
    seed_engine_stream,
)

ALGORITHMS = (
    MonteCarloAgent,
    QLearningAgent,
    SarsaAgent,
    DoubleQLearningAgent,
)


class _DealSpy(CallingPlayer):
    """Records the cards dealt at the start of each hand."""

    def __init__(self, log, **kwargs):
        super().__init__(**kwargs)
        self.log = log

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.log.append((round_count, tuple(hole_card)))


class _BurningPlayer(PlayerTemplate):
    """Always calls, and burns a configurable number of draws per decision.

    Two instances with different ``burn`` values make byte-for-byte identical
    decisions, so any card difference between them is a pure random-stream
    artefact rather than a consequence of play.
    """

    def __init__(self, burn, log, rng=None, **kwargs):
        super().__init__(**kwargs)
        self.burn = burn
        self.log = log
        self.rng = rng if rng is not None else random

    def declare_action(self, valid_actions, hole_card, round_state):
        for _ in range(self.burn):
            self.rng.random()

        call = next(item for item in valid_actions if item["action"] == "call")
        return call["action"], call["amount"]

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.log.append((round_count, tuple(hole_card)))

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def _play_with_burning_agent(*, burn, game_seed, isolated, rounds=8):
    log = []
    streams = derive_game_streams(game_seed)

    if isolated:
        seed_engine_stream(streams.deck_seed)
        agent_rng = streams.agent
    else:
        # The pre-fix arrangement: everyone shares the global stream.
        random.seed(game_seed)
        agent_rng = random

    config = setup_config(max_round=rounds, initial_stack=1000, small_blind_amount=5)
    config.register_player(
        name="tested",
        algorithm=_BurningPlayer(burn, log, rng=agent_rng, player_name="tested"),
    )
    config.register_player(
        name="opponent",
        algorithm=_BurningPlayer(0, [], rng=agent_rng, player_name="opponent"),
    )
    start_poker(config, verbose=0)
    return log


def test_agent_draw_count_does_not_change_the_deck():
    """The core guard: burning draws must not move the deck.

    On a shared stream one extra ``random.random()`` per decision shifts every
    later shuffle in the game, which silently defeats common random numbers.
    """
    baseline = _play_with_burning_agent(burn=0, game_seed=2024, isolated=True)

    for burn in (1, 2, 5):
        perturbed = _play_with_burning_agent(
            burn=burn,
            game_seed=2024,
            isolated=True,
        )
        assert perturbed == baseline, (
            f"burning {burn} draw(s) per decision changed the cards"
        )


def test_shared_stream_is_what_breaks_the_deck():
    """Pins the mechanism the fix removes, so the guard above cannot rot.

    If this ever stops failing to match, the engine has stopped drawing from the
    global module and the isolation above is no longer doing any work.
    """
    baseline = _play_with_burning_agent(burn=0, game_seed=2024, isolated=False)
    perturbed = _play_with_burning_agent(burn=1, game_seed=2024, isolated=False)

    assert perturbed != baseline


def _play_algorithm(agent_class, *, training_seed=555, episodes=3, rounds=6):
    log = []
    agent = agent_class()

    for episode_index in range(episodes):
        streams = derive_episode_streams(training_seed, episode_index)
        seed_engine_stream(streams.deck_seed)

        config = setup_config(
            max_round=rounds,
            initial_stack=1000,
            small_blind_amount=5,
        )
        player = GeneralPolicyPlayer(agent=agent, player_name="tested", verbose=False)
        player.training = True
        opponent = _DealSpy(log, player_name="opponent", rng=streams.opponent)

        attach_rng(player, streams.agent)
        attach_rng(opponent, streams.opponent)

        config.register_player(name="tested", algorithm=player)
        config.register_player(name="opponent", algorithm=opponent)
        start_poker(config, verbose=0)

        log.append(("episode-boundary", episode_index))

    return log


def test_all_algorithms_are_dealt_the_same_hands():
    """The confound this PR exists to remove.

    The algorithms consume different numbers of draws - Double Q-learning flips
    a coin on every update - so on a shared stream each one trains on a
    different sequence of hands, and part of any measured difference between
    them is just luck of the deal.
    """
    deals = {
        algorithm.__name__: _play_algorithm(algorithm) for algorithm in ALGORITHMS
    }

    reference_name, reference = next(iter(deals.items()))

    for name, dealt in deals.items():
        assert dealt == reference, (
            f"{name} was dealt different hands than {reference_name}"
        )


def test_compared_agents_receive_identical_cards():
    """The common-random-numbers property build_paired_evaluation_seed promises.

    Two different tested agents facing the same opponent on the same game seed
    must see the same cards, otherwise paired comparisons carry deck noise.
    """
    monte_carlo = _play_algorithm(MonteCarloAgent, episodes=2)
    sarsa = _play_algorithm(SarsaAgent, episodes=2)

    assert monte_carlo == sarsa


@pytest.mark.parametrize("opponent_type", TRAINING_OPPONENT_TYPES)
def test_every_scripted_opponent_accepts_an_injected_rng(opponent_type):
    private = random.Random(17)
    opponent = build_opponent(opponent_type, rng=private)

    assert opponent.rng is private


@pytest.mark.parametrize("opponent_type", TRAINING_OPPONENT_TYPES)
def test_scripted_opponents_default_to_the_global_stream(opponent_type):
    assert build_opponent(opponent_type).rng is random


@pytest.mark.parametrize("agent_class", ALGORITHMS)
def test_every_agent_exposes_an_rng_defaulting_to_the_global_stream(agent_class):
    assert agent_class().rng is random


def test_derive_game_streams_is_reproducible():
    first = derive_game_streams(123)
    second = derive_game_streams(123)

    assert first.deck_seed == second.deck_seed
    assert first.agent.random() == second.agent.random()
    assert first.opponent.random() == second.opponent.random()


def test_derive_game_streams_separates_the_three_sources():
    streams = derive_game_streams(4242)

    values = {
        streams.deck_seed,
        streams.agent.getrandbits(64),
        streams.opponent.getrandbits(64),
    }

    assert len(values) == 3


def test_distinct_game_seeds_produce_distinct_streams():
    first = derive_game_streams(1)
    second = derive_game_streams(2)

    assert first.deck_seed != second.deck_seed
    assert first.agent.random() != second.agent.random()


def test_episode_streams_depend_on_both_seed_and_episode():
    baseline = derive_episode_streams(7, 0)

    assert derive_episode_streams(7, 0).deck_seed == baseline.deck_seed
    assert derive_episode_streams(7, 1).deck_seed != baseline.deck_seed
    assert derive_episode_streams(8, 0).deck_seed != baseline.deck_seed


@pytest.mark.parametrize("seed", [-1, -100])
def test_derive_game_streams_rejects_a_negative_seed(seed):
    with pytest.raises(ValueError):
        derive_game_streams(seed)


@pytest.mark.parametrize("seed", [1.5, "3", None, True])
def test_derive_game_streams_rejects_a_non_integer_seed(seed):
    with pytest.raises(TypeError):
        derive_game_streams(seed)


def test_derive_episode_streams_rejects_a_negative_episode_index():
    with pytest.raises(ValueError):
        derive_episode_streams(1, -1)


def test_attach_rng_reaches_a_single_owned_agent():
    player = GeneralPolicyPlayer(
        agent=MonteCarloAgent(),
        player_name="tested",
        verbose=False,
    )
    private = random.Random(5)

    attach_rng(player, private)

    assert player.rng is private
    assert player.agent.rng is private


def test_attach_rng_reaches_every_agent_in_a_mapping():
    class _MultiAgentPlayer:
        def __init__(self):
            self.agents = {name: MonteCarloAgent() for name in ("a", "b", "c")}

    player = _MultiAgentPlayer()
    private = random.Random(5)

    attach_rng(player, private)

    assert player.rng is private
    assert all(agent.rng is private for agent in player.agents.values())


def test_attach_rng_is_harmless_for_a_player_without_agents():
    player = CallingPlayer(player_name="opponent")
    private = random.Random(5)

    attach_rng(player, private)

    assert player.rng is private
