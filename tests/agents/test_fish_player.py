from src.agents.fish_player import FishPlayer
from pypokerengine.engine.poker_constants import PokerConstants as Const


def test_declare_action_strong_preflop(valid_actions_conf):
    hole_cards = ["H9", "S9"]
    round_state = {"street": Const.Street.PREFLOP}
    valid_actions = valid_actions_conf
    fish = FishPlayer()
    action, _ = fish.declare_action(valid_actions, hole_cards, round_state)
    assert action == "call"

def test_declare_action_medium_preflop(valid_actions_conf):
    hole_cards = ['H7', 'S7']
    round_state = {"street": Const.Street.PREFLOP}
    valid_actions = valid_actions_conf
    fish = FishPlayer()
    action, _ = fish.declare_action(valid_actions,hole_cards,round_state)
    assert action == 'fold'

def test_declare_action_medium_flop(valid_actions_conf):
    pass

def test_declare_action_medium_turn(valid_actions_conf):
    pass

def test_declare_action_medium_river(valid_actions_conf):
    pass