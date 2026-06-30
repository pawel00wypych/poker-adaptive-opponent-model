from src.features.opponent_stats import OpponentStats
from src.opponent_model.rule_based_classifier import RuleBasedOpponentClassifier


def build_stats(actions: list[str]) -> OpponentStats:
    stats = OpponentStats()

    for action in actions:
        stats.update_action(action)

    return stats


def test_classifier_returns_unknown_when_not_enough_actions():
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["raise", "raise", "call"])

    assert classifier.classify(stats) == "unknown"


def test_classifier_detects_aggressive_opponent():
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["raise", "raise", "raise", "call", "fold"])

    assert classifier.classify(stats) == "aggressive"


def test_classifier_detects_fish_opponent():
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["call", "call", "call", "call", "fold"])

    assert classifier.classify(stats) == "fish"


def test_classifier_detects_tight_opponent():
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["fold", "fold", "fold", "fold", "call"])

    assert classifier.classify(stats) == "tight"


def test_classifier_detects_balanced_opponent():
    classifier = RuleBasedOpponentClassifier(min_actions=6)
    stats = build_stats(["fold", "fold", "call", "call", "raise", "call"])

    assert classifier.classify(stats) == "balanced"