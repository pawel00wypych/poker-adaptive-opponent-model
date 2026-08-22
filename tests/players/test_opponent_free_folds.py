"""Scripted opponents must never surrender a hand that costs nothing to keep."""

import random

import pytest
from pypokerengine.api.game import setup_config, start_poker

from src.config import GameConfig
from src.players.base.player_template import PlayerTemplate
from src.players.baselines.rule_based_player import RuleBasedPlayer
from src.players.generalization.generalization_opponents import (
    build_generalization_opponent_player,
)
from src.players.opponents.factory import build_opponent
from src.poker.betting import avoid_free_fold, call_cost

OPPONENT_BUILDERS = {
    "tight": lambda: build_opponent("tight"),
    "aggressive": lambda: build_opponent("aggressive"),
    "calling": lambda: build_opponent("calling"),
    "rule_based": RuleBasedPlayer,
    "tight_extreme": lambda: build_generalization_opponent_player(
        "tight_extreme", rng=random.Random(3)
    ),
    "aggressive_extreme": lambda: build_generalization_opponent_player(
        "aggressive_extreme", rng=random.Random(3)
    ),
    "calling_extreme": lambda: build_generalization_opponent_player(
        "calling_extreme", rng=random.Random(3)
    ),
}


class ContinuingPlayer(PlayerTemplate):
    """Always calls, so the opponent faces many free decisions."""

    def declare_action(self, valid_actions, hole_card, round_state):
        call = next((a for a in valid_actions if a["action"] == "call"), None)
        if call:
            return call["action"], call["amount"]
        return "fold", 0

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.update_tracking_after_round(
            current_stack=self.get_my_stack_from_round_state(round_state)
        )


def _count_free_folds(opponent_name, games=12):
    counts = {"free": 0, "folded": 0}

    for game_index in range(games):
        random.seed(6000 + game_index)
        opponent = OPPONENT_BUILDERS[opponent_name]()
        original = opponent.declare_action

        def recording(valid_actions, hole_card, round_state, _original=original):
            free = (
                call_cost(valid_actions, round_state, opponent.player_uuid) == 0
                and any(a["action"] == "call" for a in valid_actions)
            )
            action, amount = _original(valid_actions, hole_card, round_state)

            if free:
                counts["free"] += 1
                if action == "fold":
                    counts["folded"] += 1

            return action, amount

        opponent.declare_action = recording

        config = GameConfig()
        table = setup_config(
            max_round=config.max_round,
            initial_stack=config.initial_stack,
            small_blind_amount=config.small_blind_amount,
        )
        table.register_player(name="continuing", algorithm=ContinuingPlayer())
        table.register_player(name=opponent_name, algorithm=opponent)
        start_poker(table, verbose=0)

    return counts


@pytest.mark.parametrize("opponent_name", sorted(OPPONENT_BUILDERS))
def test_opponent_never_folds_when_staying_is_free(opponent_name):
    """Before the fix, tight folded 44% of these spots and tight_extreme 53%."""
    counts = _count_free_folds(opponent_name)

    assert counts["free"] > 0, "the probe produced no free decisions"
    assert counts["folded"] == 0, counts


def test_avoid_free_fold_converts_a_free_fold_into_a_check():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
    ]
    round_state = {
        "street": "preflop",
        "action_histories": {
            "preflop": [
                {"action": "BIGBLIND", "amount": 10, "uuid": "uuid-bb"},
            ]
        },
    }

    action, amount = avoid_free_fold(
        "fold", 0, valid_actions, round_state, "uuid-bb"
    )

    assert action == "call"
    # The engine receives the level and subtracts what was already paid.
    assert amount == 10


def test_avoid_free_fold_leaves_a_paid_fold_alone():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 40},
    ]
    round_state = {"street": "flop", "action_histories": {"flop": []}}

    assert avoid_free_fold(
        "fold", 0, valid_actions, round_state, "uuid-bb"
    ) == ("fold", 0)


def test_avoid_free_fold_leaves_other_actions_alone():
    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 0},
    ]
    round_state = {"street": "flop", "action_histories": {"flop": []}}

    assert avoid_free_fold(
        "raise", 20, valid_actions, round_state, "uuid-bb"
    ) == ("raise", 20)


def test_tight_player_judges_cost_not_the_bet_level():
    """The big blind faces a level of 10 having already posted 10."""
    from src.players.opponents.tight_player import TightPlayer

    player = TightPlayer(rng=random.Random(1))
    player.uuid = "uuid-bb"

    round_state = {
        "street": "preflop",
        "seats": [
            {"name": "bb", "uuid": "uuid-bb", "stack": 190, "state": "participating"},
        ],
        "community_card": [],
        "action_histories": {
            "preflop": [
                {"action": "BIGBLIND", "amount": 10, "uuid": "uuid-bb"},
            ]
        },
    }

    # A weak hand would otherwise be folded with high probability.
    for _ in range(20):
        action, _ = player.declare_action(
            [
                {"action": "fold", "amount": 0},
                {"action": "call", "amount": 10},
            ],
            ["H7", "D2"],
            round_state,
        )
        assert action == "call"
