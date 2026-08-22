"""Real cost of calling versus the bet level reported by the engine.

PyPokerEngine reports ``call`` as the level to match and subtracts what the
player already paid on the street. Reading that field as a cost is wrong for
anyone already invested in the street, most visibly the big blind before the
flop.
"""

import pytest

from src.poker.betting import (
    call_cost,
    is_free_check,
    paid_this_street,
    to_decision_actions,
)

SMALL_BLIND_UUID = "uuid-sb"
BIG_BLIND_UUID = "uuid-bb"


def preflop_round_state():
    return {
        "street": "preflop",
        "action_histories": {
            "preflop": [
                {
                    "action": "SMALLBLIND",
                    "amount": 5,
                    "add_amount": 5,
                    "uuid": SMALL_BLIND_UUID,
                },
                {
                    "action": "BIGBLIND",
                    "amount": 10,
                    "add_amount": 5,
                    "uuid": BIG_BLIND_UUID,
                },
            ]
        },
    }


def flop_round_state():
    return {"street": "flop", "action_histories": {"preflop": [], "flop": []}}


def valid_actions(call_amount, with_raise=True):
    actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": call_amount},
    ]
    if with_raise:
        actions.append({"action": "raise", "amount": {"min": 20, "max": 200}})
    return actions


def test_paid_this_street_reads_the_big_blind_posting():
    assert paid_this_street(preflop_round_state(), BIG_BLIND_UUID) == 10
    assert paid_this_street(preflop_round_state(), SMALL_BLIND_UUID) == 5


def test_paid_this_street_is_zero_on_a_fresh_street():
    assert paid_this_street(flop_round_state(), BIG_BLIND_UUID) == 0


def test_paid_this_street_ignores_other_streets():
    round_state = {
        "street": "flop",
        "action_histories": {
            "preflop": [
                {"action": "CALL", "amount": 40, "uuid": BIG_BLIND_UUID},
            ],
            "flop": [],
        },
    }

    assert paid_this_street(round_state, BIG_BLIND_UUID) == 0


def test_paid_this_street_ignores_folds_and_antes():
    round_state = {
        "street": "flop",
        "action_histories": {
            "flop": [
                {"action": "ANTE", "amount": 1, "uuid": BIG_BLIND_UUID},
                {"action": "FOLD", "uuid": BIG_BLIND_UUID},
            ]
        },
    }

    assert paid_this_street(round_state, BIG_BLIND_UUID) == 0


def test_paid_this_street_uses_the_latest_level():
    """History entries store cumulative levels, so the last one wins."""
    round_state = {
        "street": "flop",
        "action_histories": {
            "flop": [
                {"action": "CALL", "amount": 10, "uuid": BIG_BLIND_UUID},
                {"action": "RAISE", "amount": 40, "uuid": BIG_BLIND_UUID},
            ]
        },
    }

    assert paid_this_street(round_state, BIG_BLIND_UUID) == 40


def test_big_blind_call_is_free_before_the_flop():
    """The level reads 10, but the big blind already posted 10."""
    actions = valid_actions(10)
    round_state = preflop_round_state()

    assert call_cost(actions, round_state, BIG_BLIND_UUID) == 0
    assert is_free_check(actions, round_state, BIG_BLIND_UUID) is True


def test_small_blind_still_pays_to_complete():
    actions = valid_actions(10)
    round_state = preflop_round_state()

    assert call_cost(actions, round_state, SMALL_BLIND_UUID) == 5
    assert is_free_check(actions, round_state, SMALL_BLIND_UUID) is False


def test_call_cost_never_goes_negative():
    round_state = {
        "street": "flop",
        "action_histories": {
            "flop": [{"action": "RAISE", "amount": 80, "uuid": BIG_BLIND_UUID}]
        },
    }

    assert call_cost(valid_actions(40), round_state, BIG_BLIND_UUID) == 0


def test_call_cost_is_zero_without_a_call_action():
    actions = [{"action": "fold", "amount": 0}]

    assert call_cost(actions, flop_round_state(), BIG_BLIND_UUID) == 0
    assert is_free_check(actions, flop_round_state(), BIG_BLIND_UUID) is False


def test_decision_actions_replace_the_call_level_with_the_cost():
    decision_actions = to_decision_actions(
        valid_actions(10),
        preflop_round_state(),
        BIG_BLIND_UUID,
    )

    call_action = next(a for a in decision_actions if a["action"] == "call")

    assert call_action["amount"] == 0


def test_decision_actions_leave_raise_amounts_untouched():
    decision_actions = to_decision_actions(
        valid_actions(10),
        preflop_round_state(),
        BIG_BLIND_UUID,
    )

    raise_action = next(a for a in decision_actions if a["action"] == "raise")

    assert raise_action["amount"] == {"min": 20, "max": 200}


def test_decision_actions_do_not_mutate_the_engine_actions():
    """The reply to the engine must still carry the level."""
    actions = valid_actions(10)

    to_decision_actions(actions, preflop_round_state(), BIG_BLIND_UUID)

    call_action = next(a for a in actions if a["action"] == "call")
    assert call_action["amount"] == 10


@pytest.mark.parametrize("call_amount", [0, 5, 10, 40])
def test_cost_equals_level_on_a_fresh_street(call_amount):
    actions = valid_actions(call_amount)

    assert (
        call_cost(actions, flop_round_state(), BIG_BLIND_UUID) == call_amount
    )


def test_free_check_share_matches_the_engine_in_real_games():
    """End-to-end: the big blind's preflop option is now seen as free."""
    import random

    from pypokerengine.api.game import setup_config, start_poker

    from src.players.base.player_template import PlayerTemplate

    seen = {"decisions": 0, "free": 0, "preflop_free": 0}

    class Probe(PlayerTemplate):
        def declare_action(self, valid_actions, hole_card, round_state):
            seen["decisions"] += 1
            if is_free_check(valid_actions, round_state, self.uuid):
                seen["free"] += 1
                if round_state.get("street") == "preflop":
                    seen["preflop_free"] += 1

            call = next(a for a in valid_actions if a["action"] == "call")
            return call["action"], call["amount"]

        def receive_game_update_message(self, action, round_state):
            pass

        def receive_round_result_message(self, winners, hand_info, round_state):
            self.update_tracking_after_round(
                current_stack=self.get_my_stack_from_round_state(round_state)
            )

    random.seed(11)
    config = setup_config(max_round=20, initial_stack=200, small_blind_amount=5)
    config.register_player(name="p1", algorithm=Probe())
    config.register_player(name="p2", algorithm=Probe())
    start_poker(config, verbose=0)

    assert seen["decisions"] > 0
    # The big blind facing a limp is free even though the level reads 10.
    assert seen["preflop_free"] > 0
    assert seen["free"] > seen["preflop_free"]
