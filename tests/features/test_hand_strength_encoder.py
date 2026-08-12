import pytest

from src.features.hand_strength_encoder import (
    HandStrengthEncoder,
)
from src.features.preflop_hand_encoder import (
    PreflopHandEncoder,
)


def test_preflop_uses_preflop_encoder():
    strength = HandStrengthEncoder.encode(
        hole_cards=["HA", "DA"],
        community_cards=[],
    )

    assert strength == PreflopHandEncoder.PREMIUM


def test_postflop_detects_high_card():
    strength = HandStrengthEncoder.encode(
        hole_cards=["HA", "DK"],
        community_cards=["C2", "S7", "H9"],
    )

    assert strength == HandStrengthEncoder.HIGH_CARD


def test_postflop_detects_one_pair():
    strength = HandStrengthEncoder.encode(
        hole_cards=["HA", "DK"],
        community_cards=["CA", "S7", "H9"],
    )

    assert strength == HandStrengthEncoder.ONE_PAIR


def test_postflop_detects_two_pair():
    strength = HandStrengthEncoder.encode(
        hole_cards=["HA", "DK"],
        community_cards=["CA", "SK", "H9"],
    )

    assert strength == HandStrengthEncoder.TWO_PAIR


def test_postflop_detects_three_of_a_kind():
    strength = HandStrengthEncoder.encode(
        hole_cards=["HA", "DA"],
        community_cards=["CA", "S7", "H9"],
    )

    assert strength == HandStrengthEncoder.THREE_OF_A_KIND


def test_postflop_detects_straight():
    strength = HandStrengthEncoder.encode(
        hole_cards=["H9", "DT"],
        community_cards=["CJ", "SQ", "HK"],
    )

    assert strength == HandStrengthEncoder.STRAIGHT


def test_postflop_detects_flush():
    strength = HandStrengthEncoder.encode(
        hole_cards=["HA", "H2"],
        community_cards=["H5", "H7", "HJ"],
    )

    assert strength == HandStrengthEncoder.FLUSH


def test_postflop_detects_full_house():
    strength = HandStrengthEncoder.encode(
        hole_cards=["HA", "DA"],
        community_cards=["CA", "SK", "HK"],
    )

    assert strength == HandStrengthEncoder.FULL_HOUSE


def test_invalid_hole_card_count_raises_error():
    with pytest.raises(ValueError):
        HandStrengthEncoder.encode(
            hole_cards=["HA"],
            community_cards=[],
        )


def test_invalid_community_card_count_raises_error():
    with pytest.raises(ValueError):
        HandStrengthEncoder.encode(
            hole_cards=["HA", "DA"],
            community_cards=["C2"],
        )