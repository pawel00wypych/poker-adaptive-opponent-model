from src.cards.hand_estimator import HandStrength
from src.cards.hand_estimator import HandEstimator

def test_check_preflop_hole_card_strength_valid_very_strong():
    hole_cards = ['HA', 'HK']
    hand_estimator = HandEstimator()
    result = hand_estimator.check_preflop_hole_card_strength(hole_cards)
    assert result == HandStrength.VERY_STRONG_CARDS

def test_check_preflop_hole_card_strength_invalid_very_strong():
    hole_cards = ['HA', 'HT']
    hand_estimator = HandEstimator()
    result = hand_estimator.check_preflop_hole_card_strength(hole_cards)
    assert result != HandStrength.VERY_STRONG_CARDS

def test_check_preflop_hole_card_strength_valid_weak():
    hole_cards = ['HJ', 'S3']
    hand_estimator = HandEstimator()
    result = hand_estimator.check_preflop_hole_card_strength(hole_cards)
    assert result == HandStrength.WEAK_CARDS

def test_check_preflop_hole_card_strength_invalid_weak():
    hole_cards = ['HJ', 'SK']
    hand_estimator = HandEstimator()
    result = hand_estimator.check_preflop_hole_card_strength(hole_cards)
    assert result != HandStrength.WEAK_CARDS

def test_to_notation_suited():
    cards = ['HA', 'HK']
    cards_in_notation = HandEstimator.to_notation(cards[0], cards[1])
    assert cards_in_notation == 'AKs'

def test_to_notation_offsuit():
    cards = ['SA', 'HK']
    cards_in_notation = HandEstimator.to_notation(cards[0], cards[1])
    assert cards_in_notation != 'AKs'