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


def test_small_blind_opens_every_street():
    """Characterisation test: pins the engine's acting order.

    This test passes on unfixed code too, and that is the point - it does not
    fix anything. PyPokerEngine decides who acts first in ``__start_street``
    before branching on the street:

        next_player_pos = state["table"].next_ask_waiting_player_pos(
            state["table"].sb_pos() - 1
        )

    so the small blind opens the flop, turn and river as well as preflop. Real
    heads-up play has the big blind open postflop, which means the agent never
    experiences acting last with information.

    If this test ever fails, the environment has changed and the README section
    "No postflop positional alternation" - along with any positional claim in
    the thesis - has to be revisited.
    """
    from collections import Counter, defaultdict

    from pypokerengine.api.game import setup_config, start_poker

    from src.players.base.player_template import PlayerTemplate
    from src.poker.round_state_utils import is_small_blind
    from src.rl.rng import derive_game_streams, seed_engine_stream

    first_actor = defaultdict(Counter)
    street_state = {}

    class Probe(PlayerTemplate):
        def receive_street_start_message(self, street, round_state):
            street_state["street"] = street
            street_state["opened"] = False

        def declare_action(self, valid_actions, hole_card, round_state):
            street = street_state.get("street")

            if street is not None and not street_state.get("opened"):
                street_state["opened"] = True
                position = (
                    "small_blind"
                    if is_small_blind(round_state, self.uuid)
                    else "big_blind"
                )
                first_actor[street][position] += 1

            call = next(item for item in valid_actions if item["action"] == "call")
            return call["action"], call["amount"]

        def receive_game_update_message(self, action, round_state):
            pass

        def receive_round_result_message(self, winners, hand_info, round_state):
            pass

    for seed in range(8):
        streams = derive_game_streams(seed)
        seed_engine_stream(streams.deck_seed)

        config = setup_config(
            max_round=10,
            initial_stack=100_000,
            small_blind_amount=5,
        )
        config.register_player(name="p1", algorithm=Probe(player_name="p1"))
        config.register_player(name="p2", algorithm=Probe(player_name="p2"))
        start_poker(config, verbose=0)

    for street in ("preflop", "flop", "turn", "river"):
        observed = first_actor[street]
        assert sum(observed.values()) > 0, f"no {street} was observed"
        assert observed["big_blind"] == 0, (
            f"the big blind opened {street} {observed['big_blind']} time(s); "
            "the engine has gained positional alternation and the README "
            "limitation section is now wrong"
        )


def test_the_agent_never_acts_last_postflop():
    """States the consequence directly, in the terms the thesis has to use.

    Acting last with information is the structural advantage of position in real
    heads-up poker. In this environment the small blind never has it.
    """
    from pypokerengine.api.game import setup_config, start_poker

    from src.players.base.player_template import PlayerTemplate
    from src.poker.round_state_utils import is_small_blind
    from src.rl.rng import derive_game_streams, seed_engine_stream

    decisions = []

    class Probe(PlayerTemplate):
        def declare_action(self, valid_actions, hole_card, round_state):
            decisions.append(
                (
                    round_state.get("round_count"),
                    round_state.get("street"),
                    "small_blind"
                    if is_small_blind(round_state, self.uuid)
                    else "big_blind",
                )
            )
            call = next(item for item in valid_actions if item["action"] == "call")
            return call["action"], call["amount"]

        def receive_game_update_message(self, action, round_state):
            pass

        def receive_round_result_message(self, winners, hand_info, round_state):
            pass

    for seed in range(8):
        streams = derive_game_streams(seed)
        seed_engine_stream(streams.deck_seed)

        config = setup_config(
            max_round=10,
            initial_stack=100_000,
            small_blind_amount=5,
        )
        config.register_player(name="p1", algorithm=Probe(player_name="p1"))
        config.register_player(name="p2", algorithm=Probe(player_name="p2"))
        start_poker(config, verbose=0)

    # Last decision taken within each (hand, street) group.
    closers = {}
    for hand, street, position in decisions:
        closers[(hand, street)] = position

    postflop = {
        position
        for (_, street), position in closers.items()
        if street in {"flop", "turn", "river"}
    }

    assert postflop, "no postflop street was completed"
    assert postflop == {"big_blind"}, (
        f"postflop streets were closed by {sorted(postflop)}; real heads-up "
        "play has the small blind close them, and this engine does not"
    )
