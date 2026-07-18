import pytest

from PyPokerEngine.pypokerengine.engine.card import Card
from PyPokerEngine.pypokerengine.engine.hand_evaluator import HandEvaluator


def cards(card_strings: list[str]) -> list[Card]:
    return [
        Card.from_str(card)
        for card in card_strings
    ]


def score(
    hole_cards: list[str],
    community_cards: list[str],
) -> int:
    return HandEvaluator.eval_hand(
        cards(hole_cards),
        cards(community_cards),
    )


def hand_info(
    hole_cards: list[str],
    community_cards: list[str],
) -> dict:
    return HandEvaluator.gen_hand_rank_info(
        cards(hole_cards),
        cards(community_cards),
    )


def assert_hand_a_wins(
    hole_a: list[str],
    hole_b: list[str],
    board: list[str],
) -> None:
    assert score(hole_a, board) > score(hole_b, board)


def assert_split(
    hole_a: list[str],
    hole_b: list[str],
    board: list[str],
) -> None:
    assert score(hole_a, board) == score(hole_b, board)


def test_pair_beats_high_card():
    board = [
        "C2",
        "D7",
        "S9",
        "HJ",
        "CK",
    ]

    pair_hand = [
        "HA",
        "DA",
    ]

    high_card_hand = [
        "S8",
        "D3",
    ]

    assert_hand_a_wins(
        pair_hand,
        high_card_hand,
        board,
    )


def test_flush_beats_straight():
    board = [
        "H2",
        "H5",
        "H9",
        "C8",
        "DT",
    ]

    flush_hand = [
        "HA",
        "HK",
    ]

    straight_hand = [
        "S6",
        "D7",
    ]

    assert_hand_a_wins(
        flush_hand,
        straight_hand,
        board,
    )


def test_full_house_beats_flush():
    board = [
        "HA",
        "DA",
        "C2",
        "H7",
        "H9",
    ]

    full_house_hand = [
        "SA",
        "D2",
    ]

    flush_hand = [
        "HK",
        "HQ",
    ]

    assert_hand_a_wins(
        full_house_hand,
        flush_hand,
        board,
    )


def test_higher_full_house_wins():
    board = [
        "HA",
        "DA",
        "CK",
        "DK",
        "S2",
    ]

    aces_full = [
        "SA",
        "C2",
    ]

    kings_full = [
        "SK",
        "D2",
    ]

    assert_hand_a_wins(
        aces_full,
        kings_full,
        board,
    )


def test_four_of_a_kind_beats_full_house():
    board = [
        "HA",
        "DA",
        "CA",
        "CK",
        "D2",
    ]

    quads = [
        "SA",
        "H3",
    ]

    full_house = [
        "SK",
        "DK",
    ]

    assert_hand_a_wins(
        quads,
        full_house,
        board,
    )


def test_straight_flush_beats_four_of_a_kind():
    board = [
        "H9",
        "HT",
        "HJ",
        "HQ",
        "CA",
    ]

    straight_flush = [
        "HK",
        "D2",
    ]

    quads = [
        "SA",
        "DA",
    ]

    assert_hand_a_wins(
        straight_flush,
        quads,
        board,
    )


def test_higher_pair_kicker_wins_when_hole_card_matters():
    board = [
        "CA",
        "D7",
        "S4",
        "H2",
        "C9",
    ]

    ace_king = [
        "SA",
        "DK",
    ]

    ace_queen = [
        "HA",
        "CQ",
    ]

    assert_hand_a_wins(
        ace_king,
        ace_queen,
        board,
    )


def test_hand_info_reports_expected_strength():
    info = hand_info(
        hole_cards=[
            "HA",
            "DA",
        ],
        community_cards=[
            "CA",
            "D7",
            "S4",
            "H2",
            "C9",
        ],
    )

    assert (
        info["hand"]["strength"]
        == "THREECARD"
    )

    assert info["hand"]["high"] == 14


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PyPokerEngine appears to use hole-card bits as tie-breakers "
        "even when the best five-card hand is entirely on the board."
    ),
)
def test_board_only_straight_should_split():
    board = [
        "C9",
        "DT",
        "HJ",
        "SQ",
        "CK",
    ]

    first = [
        "SA",
        "D2",
    ]

    second = [
        "H8",
        "C7",
    ]

    assert_split(
        first,
        second,
        board,
    )


def test_wheel_straight_should_beat_high_card():
    board = [
        "C2",
        "D3",
        "H4",
        "S9",
        "DT",
    ]

    wheel = [
        "SA",
        "H5",
    ]

    high_card = [
        "HK",
        "DQ",
    ]

    assert_hand_a_wins(
        wheel,
        high_card,
        board,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PyPokerEngine appears to break ties using hole cards even "
        "when a board-only flush should split the pot."
    ),
)
def test_board_only_flush_should_split():
    board = [
        "HA",
        "HK",
        "HQ",
        "HJ",
        "H9",
    ]

    first = [
        "SA",
        "D2",
    ]

    second = [
        "C8",
        "D7",
    ]

    assert_split(
        first,
        second,
        board,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PyPokerEngine appears to break ties using hole cards even "
        "when board-only two pair with board kicker should split."
    ),
)
def test_board_only_two_pair_with_board_kicker_should_split():
    board = [
        "HA",
        "DA",
        "CK",
        "DK",
        "SQ",
    ]

    first = [
        "SJ",
        "DT",
    ]

    second = [
        "C9",
        "D8",
    ]

    assert_split(
        first,
        second,
        board,
    )