import pytest

from src.features.hand_strength_encoder import (
    HandStrengthEncoder,
)
from src.features.poker_context_encoder import (
    PokerContextEncoder,
)


def round_state_with_pot(amount: int) -> dict:
    return {
        "pot": {
            "main": {
                "amount": amount,
            }
        }
    }


def valid_actions_with_call(amount: int) -> list[dict]:
    return [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": amount,
        },
        {
            "action": "raise",
            "amount": {
                "min": 20,
                "max": 200,
            },
        },
    ]


def test_pair_strength_preflop_has_no_pair_context():
    bucket = PokerContextEncoder.pair_strength_bucket(
        hole_cards=["HA", "DA"],
        community_cards=[],
        hand_strength_bin=HandStrengthEncoder.ONE_PAIR,
    )

    assert (
        bucket
        == PokerContextEncoder.NO_PAIR_CONTEXT
    )


def test_pair_strength_high_card_has_no_pair_context():
    bucket = PokerContextEncoder.pair_strength_bucket(
        hole_cards=["H9", "D8"],
        community_cards=["CA", "SK", "D2"],
        hand_strength_bin=HandStrengthEncoder.HIGH_CARD,
    )

    assert (
        bucket
        == PokerContextEncoder.NO_PAIR_CONTEXT
    )


def test_pair_strength_two_pair_or_better():
    bucket = PokerContextEncoder.pair_strength_bucket(
        hole_cards=["HA", "DK"],
        community_cards=["CA", "SK", "D2"],
        hand_strength_bin=HandStrengthEncoder.TWO_PAIR,
    )

    assert (
        bucket
        == PokerContextEncoder.TWO_PAIR_OR_BETTER
    )


def test_pair_strength_overpair():
    bucket = PokerContextEncoder.pair_strength_bucket(
        hole_cards=["HA", "DA"],
        community_cards=["CK", "S7", "D2"],
        hand_strength_bin=HandStrengthEncoder.ONE_PAIR,
    )

    assert (
        bucket
        == PokerContextEncoder.OVERPAIR
    )


def test_pair_strength_underpair_for_pocket_pair_below_board():
    bucket = PokerContextEncoder.pair_strength_bucket(
        hole_cards=["H2", "D2"],
        community_cards=["CA", "SK", "DQ"],
        hand_strength_bin=HandStrengthEncoder.ONE_PAIR,
    )

    assert (
        bucket
        == PokerContextEncoder.UNDER_PAIR
    )


def test_pair_strength_top_pair():
    bucket = PokerContextEncoder.pair_strength_bucket(
        hole_cards=["HA", "D9"],
        community_cards=["CA", "SK", "D2"],
        hand_strength_bin=HandStrengthEncoder.ONE_PAIR,
    )

    assert (
        bucket
        == PokerContextEncoder.TOP_PAIR
    )


def test_pair_strength_middle_pair():
    bucket = PokerContextEncoder.pair_strength_bucket(
        hole_cards=["HK", "D9"],
        community_cards=["CA", "SK", "D2"],
        hand_strength_bin=HandStrengthEncoder.ONE_PAIR,
    )

    assert (
        bucket
        == PokerContextEncoder.MIDDLE_PAIR
    )


def test_pair_strength_bottom_pair():
    bucket = PokerContextEncoder.pair_strength_bucket(
        hole_cards=["H2", "D9"],
        community_cards=["CA", "SK", "D2"],
        hand_strength_bin=HandStrengthEncoder.ONE_PAIR,
    )

    assert (
        bucket
        == PokerContextEncoder.BOTTOM_PAIR
    )


def test_pair_strength_underpair_when_pair_is_only_on_board():
    bucket = PokerContextEncoder.pair_strength_bucket(
        hole_cards=["H9", "D8"],
        community_cards=["CA", "SA", "D2"],
        hand_strength_bin=HandStrengthEncoder.ONE_PAIR,
    )

    assert (
        bucket
        == PokerContextEncoder.UNDER_PAIR
    )


def test_pot_odds_bucket_free_check():
    bucket = PokerContextEncoder.pot_odds_bucket(
        valid_actions=valid_actions_with_call(0),
        round_state=round_state_with_pot(100),
    )

    assert bucket == 0


def test_pot_odds_bucket_without_call_action():
    bucket = PokerContextEncoder.pot_odds_bucket(
        valid_actions=[
            {
                "action": "fold",
                "amount": 0,
            },
            {
                "action": "raise",
                "amount": {
                    "min": 20,
                    "max": 200,
                },
            },
        ],
        round_state=round_state_with_pot(100),
    )

    assert bucket == 0


@pytest.mark.parametrize(
    ("pot", "call", "expected_bucket"),
    [
        (100, 10, 1),   # 10 / 110 = 0.091
        (100, 30, 2),   # 30 / 130 = 0.231
        (100, 70, 3),   # 70 / 170 = 0.412
        (100, 100, 4),  # 100 / 200 = 0.500
    ],
)
def test_pot_odds_buckets(
    pot,
    call,
    expected_bucket,
):
    bucket = PokerContextEncoder.pot_odds_bucket(
        valid_actions=valid_actions_with_call(call),
        round_state=round_state_with_pot(pot),
    )

    assert bucket == expected_bucket


@pytest.mark.parametrize(
    ("stack", "pot", "expected_bucket"),
    [
        (100, 0, 3),
        (100, 100, 0),  # SPR = 1
        (150, 100, 1),  # SPR = 1.5
        (500, 100, 2),  # SPR = 5
        (700, 100, 3),  # SPR = 7
    ],
)
def test_spr_buckets(
    stack,
    pot,
    expected_bucket,
):
    bucket = PokerContextEncoder.spr_bucket(
        player_stack=stack,
        round_state=round_state_with_pot(pot),
    )

    assert bucket == expected_bucket

def test_pot_odds_use_the_real_call_cost_for_the_big_blind():
    """Regression for reading the engine's bet level as a cost.

    The big blind faces a level of 10 having already posted 10, so calling is
    free and pot odds are zero. Reading the raw level gave 10 / (15 + 10) = 0.4
    and put the state in a completely different bucket.
    """
    from src.poker.betting import to_decision_actions

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
        {"action": "raise", "amount": {"min": 20, "max": 200}},
    ]
    round_state = {
        "street": "preflop",
        "community_card": [],
        "pot": {"main": {"amount": 15}},
        "action_histories": {
            "preflop": [
                {"action": "SMALLBLIND", "amount": 5, "uuid": "uuid-sb"},
                {"action": "BIGBLIND", "amount": 10, "uuid": "uuid-bb"},
            ]
        },
    }

    raw_bucket = PokerContextEncoder.pot_odds_bucket(
        valid_actions=valid_actions,
        round_state=round_state,
    )

    decision_actions = to_decision_actions(valid_actions, round_state, "uuid-bb")
    corrected_bucket = PokerContextEncoder.pot_odds_bucket(
        valid_actions=decision_actions,
        round_state=round_state,
    )

    assert raw_bucket == 3
    assert corrected_bucket == 0


def test_pot_odds_unchanged_for_a_player_with_nothing_invested():
    from src.poker.betting import to_decision_actions

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
    ]
    round_state = {
        "street": "flop",
        "community_card": ["HA", "D7", "C2"],
        "pot": {"main": {"amount": 40}},
        "action_histories": {"flop": []},
    }

    decision_actions = to_decision_actions(valid_actions, round_state, "uuid-bb")

    assert PokerContextEncoder.pot_odds_bucket(
        valid_actions=decision_actions,
        round_state=round_state,
    ) == PokerContextEncoder.pot_odds_bucket(
        valid_actions=valid_actions,
        round_state=round_state,
    )
