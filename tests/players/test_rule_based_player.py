import pytest
from pypokerengine.api.game import setup_config, start_poker

from src.config import GameConfig
from src.players.baselines.rule_based_player import RuleBasedPlayer
from src.players.opponents.aggressive_player import AggressivePlayer
from src.rl.rng import attach_rng, derive_game_streams, seed_engine_stream


def _game_info(small_blind_amount=5):
    return {
        "player_num": 2,
        "rule": {
            "initial_stack": 1000,
            "max_round": 12,
            "small_blind_amount": small_blind_amount,
            "ante": 0,
            "blind_structure": {},
        },
        "seats": [],
    }


def test_rule_based_player_calls_when_call_is_free():
    player = RuleBasedPlayer()

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 0},
        {"action": "raise", "amount": {"min": 10, "max": 100}},
    ]

    action, amount = player.declare_action(valid_actions, hole_card=[], round_state={})

    assert action == "call"
    assert amount == 0


def test_rule_based_player_folds_expensive_call():
    player = RuleBasedPlayer()

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 50},
        {"action": "raise", "amount": {"min": 80, "max": 100}},
    ]

    action, amount = player.declare_action(valid_actions, hole_card=[], round_state={})

    assert action == "fold"
    assert amount == 0


def test_rule_based_player_calls_cheap_call():
    player = RuleBasedPlayer()

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    action, amount = player.declare_action(valid_actions, hole_card=[], round_state={})

    assert action == "call"
    assert amount == 10

def test_rule_based_player_tracks_round_results():
    player = RuleBasedPlayer(player_name="rule_based")
    player.uuid = "uuid-rule"

    round_state_1 = {
        "seats": [
            {"name": "rule_based", "uuid": "uuid-rule", "stack": 100},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 100},
        ]
    }

    round_state_2 = {
        "seats": [
            {"name": "rule_based", "uuid": "uuid-rule", "stack": 70},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 130},
        ]
    }

    player.receive_round_result_message([], [], round_state_1)
    player.receive_round_result_message([], [], round_state_2)

    assert player.hands_played == 2
    assert player.total_reward_bb == -3.0


def test_action_counter_resets_on_game_start():
    player = RuleBasedPlayer()
    player.action_counter = 7

    player.receive_game_start_message(_game_info())

    assert player.action_counter == 0


def _record_one_game(player, seed=4242, small_blind_amount=5, rounds=9):
    """Play one game with a fixed seed and return the decisions taken.

    Two choices here are load-bearing, and the guard test below pins both:

    * The opponent is ``AggressivePlayer``. Against a calling opponent every
      decision is a free check, which short-circuits before the raise cadence
      is ever consulted - so a leaking counter has no observable effect and the
      test would pass on broken code.
    * ``rounds=9`` avoids a decision count that is a multiple of the
      10-decision cadence, which would land the next game on the same offset
      and hide the leak just as effectively.
    """
    decisions = []
    original = player.declare_action

    def recording(valid_actions, hole_card, round_state):
        decision = original(valid_actions, hole_card, round_state)
        decisions.append(decision)
        return decision

    player.declare_action = recording
    try:
        streams = derive_game_streams(seed)
        seed_engine_stream(streams.deck_seed)

        config = setup_config(
            max_round=rounds,
            initial_stack=1000,
            small_blind_amount=small_blind_amount,
        )
        opponent = AggressivePlayer(player_name="opponent", rng=streams.opponent)
        attach_rng(player, streams.agent)
        attach_rng(opponent, streams.opponent)

        config.register_player(name="rule_based", algorithm=player)
        config.register_player(name="opponent", algorithm=opponent)
        start_poker(config, verbose=0)
    finally:
        player.declare_action = original

    return decisions


def test_the_probe_game_can_actually_expose_a_leaking_counter():
    """Guards the guard.

    If either condition stops holding, the two tests below silently stop
    testing anything and pass on broken code.
    """
    decisions = _record_one_game(RuleBasedPlayer(player_name="rule_based"))
    actions = [action for action, _ in decisions]

    assert "raise" in actions, "cadence never fires, so a leak cannot be observed"
    assert len(decisions) % 10 != 0, "decision count hides the leak"


def test_two_consecutive_games_produce_identical_action_sequences():
    """The property that actually matters: the baseline is a fixed strategy.

    It is the reference point for delta_vs_rule_based, so if its behaviour
    depends on how many hands it played earlier, that comparison partly
    measures the order games were run in.
    """
    player = RuleBasedPlayer(player_name="rule_based")

    first = _record_one_game(player)
    second = _record_one_game(player)
    third = _record_one_game(player)

    assert first, "the probe recorded no decisions"
    assert second == first
    assert third == first


def test_a_reused_instance_matches_a_fresh_one():
    reused = RuleBasedPlayer(player_name="rule_based")
    _record_one_game(reused)
    after_reuse = _record_one_game(reused)

    fresh = _record_one_game(RuleBasedPlayer(player_name="rule_based"))

    assert after_reuse == fresh


@pytest.mark.parametrize(
    ("small_blind_amount", "expected_threshold"),
    [(5, 30), (10, 60), (25, 150)],
)
def test_fold_threshold_follows_the_blind_structure(
    small_blind_amount, expected_threshold
):
    player = RuleBasedPlayer()
    player.receive_game_start_message(_game_info(small_blind_amount))

    assert player.fold_threshold == expected_threshold


def test_fold_threshold_is_unchanged_at_the_default_blinds():
    """Expressing the threshold in big blinds must not move the baseline.

    3 * big_blind is exactly the literal 30 it replaces at the default
    small_blind_amount=5, so no previously recorded result changes.
    """
    assert RuleBasedPlayer().fold_threshold == 30
    assert 3 * GameConfig().big_blind_amount == 30


def test_a_call_below_the_scaled_threshold_is_still_called():
    """At SB=25 a 50-chip call is cheap; the old literal 30 would have folded it."""
    player = RuleBasedPlayer()
    player.receive_game_start_message(_game_info(25))

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 50},
        {"action": "raise", "amount": {"min": 100, "max": 1000}},
    ]

    action, amount = player.declare_action(valid_actions, hole_card=[], round_state={})

    assert action == "call"
    assert amount == 50


def test_a_call_at_the_scaled_threshold_is_folded():
    player = RuleBasedPlayer()
    player.receive_game_start_message(_game_info(25))

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 150},
        {"action": "raise", "amount": {"min": 300, "max": 1000}},
    ]

    action, _ = player.declare_action(valid_actions, hole_card=[], round_state={})

    assert action == "fold"


def test_raise_cadence_restarts_with_every_game():
    """The tenth decision *of each game* raises, not the tenth ever taken.

    The first game deliberately takes 7 decisions - not a multiple of 10 - so
    that a leaking counter would put the second game's raise at index 2 instead
    of index 9.
    """
    player = RuleBasedPlayer()
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 100}},
    ]

    def play(decision_count):
        player.receive_game_start_message(_game_info())
        return [
            player.declare_action(valid_actions, hole_card=[], round_state={})[0]
            for _ in range(decision_count)
        ]

    play(7)
    second_game = play(10)

    assert second_game[9] == "raise"
    assert second_game[:9] == ["call"] * 9
