import pytest

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.config import GameConfig
from src.players.base.player_template import PlayerTemplate
from src.players.baselines.always_call_player import AlwaysCallPlayer
from src.players.baselines.always_raise_player import AlwaysRaisePlayer
from src.players.baselines.rule_based_player import RuleBasedPlayer
from src.players.generalization.aggressive_extreme_player import (
    AggressiveExtremePlayer,
)
from src.players.generalization.calling_extreme_player import CallingExtremePlayer
from src.players.generalization.tight_extreme_player import TightExtremePlayer
from src.players.learned.adaptive_player import AdaptivePlayer
from src.players.learned.fixed_policy_player import FixedPolicyPlayer
from src.players.learned.general_policy_player import GeneralPolicyPlayer
from src.players.learned.oracle_player import OraclePlayer
from src.players.learned.specialist_policy_player import SpecialistPolicyPlayer
from src.players.opponents.aggressive_player import AggressivePlayer
from src.players.opponents.calling_player import CallingPlayer
from src.players.opponents.tight_player import TightPlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
    OPPONENT_TYPE_UNKNOWN,
)


class DummyTrackedPlayer(PlayerTemplate):
    def __init__(self):
        super().__init__(player_name="dummy_tracked")
        self.reset_tracking()


def test_tracking_initial_state():
    player = DummyTrackedPlayer()

    assert player.hands_played == 0
    assert player.total_reward_bb == 0.0
    assert player.hand_start_stack is None
    assert player.initial_stack is None


def test_tracking_first_round_has_zero_reward():
    player = DummyTrackedPlayer()

    reward_bb = player.update_tracking_after_round(current_stack=100)

    assert reward_bb == 0.0
    assert player.total_reward_bb == 0.0
    assert player.hands_played == 1
    assert player.initial_stack == 100
    assert player.hand_start_stack == 100


def test_tracking_positive_reward():
    player = DummyTrackedPlayer()

    player.update_tracking_after_round(current_stack=100)
    reward_bb = player.update_tracking_after_round(current_stack=130)

    assert reward_bb == 3.0
    assert player.total_reward_bb == 3.0
    assert player.hands_played == 2
    assert player.hand_start_stack == 130


def test_tracking_negative_reward():
    player = DummyTrackedPlayer()

    player.update_tracking_after_round(current_stack=100)
    reward_bb = player.update_tracking_after_round(current_stack=70)

    assert reward_bb == -3.0
    assert player.total_reward_bb == -3.0
    assert player.hands_played == 2
    assert player.hand_start_stack == 70

def test_dead_position_bookkeeping_is_gone():
    """Position is derived from round_state, not tracked on the player.

    The template used to compute my_position on every street start while
    nothing ever read it.
    """
    from src.players.base.player_template import PlayerTemplate

    removed = [
        "set_my_position",
        "set_available_positions",
        "my_position",
        "available_positions",
    ]

    for name in removed:
        assert not hasattr(PlayerTemplate, name), name


def _game_info(small_blind_amount):
    return {
        "player_num": 2,
        "rule": {
            "initial_stack": 200,
            "max_round": 10,
            "small_blind_amount": small_blind_amount,
            "ante": 0,
            "blind_structure": {},
        },
        "seats": [],
    }


def test_big_blind_follows_the_blinds_of_the_running_game():
    """The engine's blind structure decides, not the GameConfig default."""
    player = DummyTrackedPlayer()

    player.receive_game_start_message(_game_info(25))

    assert player.big_blind_amount == 50


def test_big_blind_falls_back_to_the_config_default_before_game_start():
    assert DummyTrackedPlayer().big_blind_amount == GameConfig().big_blind_amount


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        {},
        {"rule": None},
        {"rule": {}},
        {"rule": {"small_blind_amount": None}},
        {"rule": {"small_blind_amount": 0}},
        {"rule": {"small_blind_amount": -5}},
        {"rule": {"small_blind_amount": True}},
        {"rule": {"small_blind_amount": "5"}},
    ],
)
def test_malformed_game_info_falls_back_instead_of_crashing(malformed):
    player = DummyTrackedPlayer()

    player.receive_game_start_message(malformed)

    assert player.big_blind_amount == GameConfig().big_blind_amount


def test_reward_bb_uses_the_blind_structure_of_the_running_game():
    """A 60-chip win is 6 BB at SB=5 but only 1.2 BB at SB=25."""
    small_blinds = DummyTrackedPlayer()
    small_blinds.receive_game_start_message(_game_info(5))
    small_blinds.update_tracking_after_round(current_stack=200)

    big_blinds = DummyTrackedPlayer()
    big_blinds.receive_game_start_message(_game_info(25))
    big_blinds.update_tracking_after_round(current_stack=200)

    assert small_blinds.update_tracking_after_round(current_stack=260) == 6.0
    assert big_blinds.update_tracking_after_round(current_stack=260) == 1.2


def test_game_config_exposes_the_big_blind():
    assert GameConfig(small_blind_amount=25).big_blind_amount == 50


CONCRETE_PLAYERS = (
    AdaptivePlayer,
    AggressiveExtremePlayer,
    AggressivePlayer,
    AlwaysCallPlayer,
    AlwaysRaisePlayer,
    CallingExtremePlayer,
    CallingPlayer,
    FixedPolicyPlayer,
    GeneralPolicyPlayer,
    OraclePlayer,
    RuleBasedPlayer,
    SpecialistPolicyPlayer,
    TightExtremePlayer,
    TightPlayer,
)


def _build(player_class):
    if player_class is AdaptivePlayer:
        agents = {
            name: MonteCarloAgent()
            for name in (
                OPPONENT_TYPE_UNKNOWN,
                OPPONENT_TYPE_TIGHT,
                OPPONENT_TYPE_AGGRESSIVE,
                OPPONENT_TYPE_CALLING,
            )
        }
        return AdaptivePlayer(agents=agents, player_name="tested")

    if player_class is OraclePlayer:
        agents = {
            name: MonteCarloAgent()
            for name in (
                OPPONENT_TYPE_UNKNOWN,
                OPPONENT_TYPE_TIGHT,
                OPPONENT_TYPE_AGGRESSIVE,
                OPPONENT_TYPE_CALLING,
            )
        }
        return OraclePlayer(
            agents=agents,
            oracle_opponent_type=OPPONENT_TYPE_TIGHT,
            player_name="tested",
        )

    if player_class is FixedPolicyPlayer:
        return FixedPolicyPlayer(
            agent=MonteCarloAgent(),
            policy_type=OPPONENT_TYPE_UNKNOWN,
            player_name="tested",
        )

    if player_class is SpecialistPolicyPlayer:
        return SpecialistPolicyPlayer(
            agent=MonteCarloAgent(),
            opponent_type=OPPONENT_TYPE_TIGHT,
            player_name="tested",
        )

    if player_class is GeneralPolicyPlayer:
        return GeneralPolicyPlayer(agent=MonteCarloAgent(), player_name="tested")

    return player_class(player_name="tested")


@pytest.mark.parametrize(
    "player_class", CONCRETE_PLAYERS, ids=lambda cls: cls.__name__
)
def test_every_player_subclass_forwards_game_start_to_the_base(player_class):
    """Guards the super() hazard permanently.

    Any override of receive_game_start_message that forgets to call super()
    silently drops the blind capture, and every BB-denominated metric for that
    player quietly falls back to the configured default.
    """
    player = _build(player_class)

    player.receive_game_start_message(_game_info(25))

    assert player.small_blind_amount == 25
    assert player.big_blind_amount == 50
