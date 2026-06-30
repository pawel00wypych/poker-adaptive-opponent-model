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
                "name": "fish",
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

    assert get_player_name(state, "uuid-2") == "fish"


def test_get_alive_players():
    state = sample_round_state()

    alive = get_alive_players(state)

    assert len(alive) == 2
    assert {player["name"] for player in alive} == {"adaptive_rl", "fish"}


def test_get_round_count():
    state = sample_round_state()

    assert get_round_count(state) == 7


def test_get_main_pot_amount():
    state = sample_round_state()

    assert get_main_pot_amount(state) == 35