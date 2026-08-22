from src.poker.round_state_utils import (
    get_alive_players,
    get_main_pot_amount,
    get_player_name,
    get_player_seat,
    get_player_stack,
    get_round_count,
)


def sample_round_state():
    return {
        "round_count": 7,
        "seats": [
            {
                "name": "adaptive_rl",
                "uuid": "uuid-1",
                "stack": 80,
                "state": "participating",
            },
            {
                "name": "tight",
                "uuid": "uuid-2",
                "stack": 120,
                "state": "participating",
            },
            {
                "name": "busted",
                "uuid": "uuid-3",
                "stack": 0,
                "state": "folded",
            },
        ],
        "pot": {
            "main": {
                "amount": 35,
            }
        },
    }


def test_get_player_seat_existing_player():
    state = sample_round_state()

    seat = get_player_seat(state, "uuid-1")

    assert seat is not None
    assert seat["name"] == "adaptive_rl"


def test_get_player_seat_missing_player():
    state = sample_round_state()

    seat = get_player_seat(state, "missing")

    assert seat is None


def test_get_player_stack():
    state = sample_round_state()

    assert get_player_stack(state, "uuid-1") == 80
    assert get_player_stack(state, "uuid-2") == 120


def test_get_player_stack_missing_player_returns_zero():
    state = sample_round_state()

    assert get_player_stack(state, "missing") == 0


def test_get_player_name():
    state = sample_round_state()

    assert get_player_name(state, "uuid-2") == "tight"


def test_get_alive_players():
    state = sample_round_state()

    alive = get_alive_players(state)

    assert len(alive) == 2
    assert {player["name"] for player in alive} == {"adaptive_rl", "tight"}


def test_get_round_count():
    state = sample_round_state()

    assert get_round_count(state) == 7


def test_get_main_pot_amount():
    state = sample_round_state()

    assert get_main_pot_amount(state) == 35

def _seats():
    return [
        {"name": "p1", "uuid": "uuid-1", "stack": 200, "state": "participating"},
        {"name": "p2", "uuid": "uuid-2", "stack": 200, "state": "participating"},
    ]


def test_is_small_blind_identifies_the_poster():
    from src.poker.round_state_utils import is_small_blind

    round_state = {"seats": _seats(), "small_blind_pos": 1, "big_blind_pos": 0}

    assert is_small_blind(round_state, "uuid-2") is True
    assert is_small_blind(round_state, "uuid-1") is False


def test_is_small_blind_follows_the_rotation():
    from src.poker.round_state_utils import is_small_blind

    first_hand = {"seats": _seats(), "small_blind_pos": 1}
    second_hand = {"seats": _seats(), "small_blind_pos": 0}

    assert is_small_blind(first_hand, "uuid-1") is False
    assert is_small_blind(second_hand, "uuid-1") is True


def test_is_small_blind_is_false_when_position_is_missing():
    from src.poker.round_state_utils import is_small_blind

    assert is_small_blind({"seats": _seats()}, "uuid-1") is False


def test_is_small_blind_is_false_for_an_unknown_player():
    from src.poker.round_state_utils import is_small_blind

    round_state = {"seats": _seats(), "small_blind_pos": 0}

    assert is_small_blind(round_state, "uuid-missing") is False


def test_position_alternates_between_hands_in_a_real_game():
    """Guards against the feature being constant, which would make it useless."""
    import random

    from pypokerengine.api.game import setup_config, start_poker

    from src.players.base.player_template import PlayerTemplate
    from src.poker.round_state_utils import is_small_blind

    observed = set()

    class Probe(PlayerTemplate):
        def declare_action(self, valid_actions, hole_card, round_state):
            observed.add(is_small_blind(round_state, self.uuid))
            call = next(a for a in valid_actions if a["action"] == "call")
            return call["action"], call["amount"]

        def receive_game_update_message(self, action, round_state):
            pass

        def receive_round_result_message(self, winners, hand_info, round_state):
            self.update_tracking_after_round(
                current_stack=self.get_my_stack_from_round_state(round_state)
            )

    random.seed(21)
    config = setup_config(max_round=6, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="p1", algorithm=Probe())
    config.register_player(name="p2", algorithm=Probe())
    start_poker(config, verbose=0)

    assert observed == {True, False}
