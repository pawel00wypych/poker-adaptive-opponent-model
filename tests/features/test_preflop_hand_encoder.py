import pytest

from src.features.preflop_hand_encoder import PreflopHandEncoder


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        (["HA", "DA"], PreflopHandEncoder.PREMIUM),
        (["HK", "DK"], PreflopHandEncoder.PREMIUM),
        (["HA", "SK"], PreflopHandEncoder.PREMIUM),
        (["HQ", "DQ"], PreflopHandEncoder.PREMIUM),
        (["HJ", "DJ"], PreflopHandEncoder.STRONG),
        (["HA", "SQ"], PreflopHandEncoder.STRONG),
        (["HT", "DT"], PreflopHandEncoder.STRONG),
        (["H8", "D8"], PreflopHandEncoder.MEDIUM),
        (["HK", "SQ"], PreflopHandEncoder.STRONG),
        (["H9", "H8"], PreflopHandEncoder.SPECULATIVE),
        (["HA", "H5"], PreflopHandEncoder.SPECULATIVE),
        (["H7", "D2"], PreflopHandEncoder.WEAK),
    ],
)
def test_preflop_hand_encoding(cards, expected):
    assert PreflopHandEncoder.encode(cards) == expected


def test_invalid_number_of_cards_returns_weak():
    assert PreflopHandEncoder.encode([]) == PreflopHandEncoder.WEAK


def test_invalid_card_format_raises_error():
    with pytest.raises(ValueError):
        PreflopHandEncoder.encode(["ACE_OF_HEARTS", "D7"])
