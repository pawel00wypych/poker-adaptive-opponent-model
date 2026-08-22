from src.features.opponent_stats import OpponentStats
from src.classifier.rule_based_classifier import RuleBasedOpponentClassifier


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


def test_classifier_detects_tight_opponent_with_moderate_fold_rate():
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["call", "call", "call", "fold", "fold"])

    assert classifier.classify(stats) == "tight"


def test_classifier_detects_tight_opponent_with_high_fold_rate():
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["fold", "fold", "fold", "fold", "call", "raise"])

    assert classifier.classify(stats) == "tight"


def test_classifier_detects_other_opponent():
    classifier = RuleBasedOpponentClassifier(min_actions=6)
    stats = build_stats(["fold", "fold", "call", "call", "raise", "call", "raise"])

    assert classifier.classify(stats) == "other"

def test_classifier_detects_calling_opponent():
    classifier = RuleBasedOpponentClassifier(min_actions=10)

    stats = build_stats([
        "call",
        "call",
        "call",
        "call",
        "call",
        "call",
        "call",
        "call",
        "call",
        "fold",
    ])

    assert classifier.classify(stats) == "calling"

def test_classifier_distinguishes_tight_from_calling():
    classifier = RuleBasedOpponentClassifier(min_actions=10)

    stats = build_stats([
        "call",
        "call",
        "call",
        "call",
        "call",
        "call",
        "raise",
        "fold",
        "fold",
        "fold",
    ])

    assert classifier.classify(stats) == "tight"