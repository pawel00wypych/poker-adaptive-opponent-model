"""Opponent classification driven by response to pressure, not raw fold rate.

Roughly three quarters of decisions in this environment are free checks. An
overall fold rate is therefore dominated by situations where folding is not
even a meaningful option, which made a tight opponent that is never bet into
look like a calling station. The classifier keys on ``fold_to_bet_rate``, and
the cases below use the ranges measured against the real scripted opponents:
tight families 0.78-0.94, calling families 0.07-0.16.
"""

from src.classifier.rule_based_classifier import RuleBasedOpponentClassifier
from src.features.opponent_stats import OpponentStats


def build_stats(actions: list[str]) -> OpponentStats:
    """Build counters from labelled actions.

    ``check`` is recorded as a call that added no chips, which is how the
    engine reports it.
    """
    stats = OpponentStats()

    for action in actions:
        if action == "check":
            stats.update_action("call", paid=0)
        elif action == "call":
            stats.update_action("call", paid=10)
        else:
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


def test_classifier_detects_tight_opponent():
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["fold", "fold", "fold", "fold", "call", "check"])

    assert classifier.classify(stats) == "tight"


def test_tight_opponent_is_recognised_despite_many_free_checks():
    """The regression that motivated the change.

    A tight player facing little aggression checks most of the time. Judged on
    overall fold rate it looks passive; judged on what it does when a bet
    arrives, it is clearly tight.
    """
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["check"] * 20 + ["fold", "fold", "fold", "call"])

    assert stats.fold_rate < 0.15
    assert stats.fold_to_bet_rate == 0.75
    assert classifier.classify(stats) == "tight"


def test_calling_opponent_is_not_confused_with_tight():
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["check"] * 20 + ["call"] * 9 + ["fold"])

    assert stats.fold_to_bet_rate == 0.1
    assert classifier.classify(stats) == "calling"


def test_classifier_detects_calling_opponent():
    classifier = RuleBasedOpponentClassifier(min_actions=10)
    stats = build_stats(["call"] * 9 + ["fold"])

    assert classifier.classify(stats) == "calling"


def test_classifier_reports_other_for_an_ambiguous_response_to_bets():
    """Half fold, half call is neither a tight nor a calling profile."""
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["call", "call", "call", "fold", "fold", "check"])

    assert stats.fold_to_bet_rate == 0.4
    assert classifier.classify(stats) == "other"


def test_classifier_waits_until_the_opponent_has_faced_a_bet():
    """Fold-to-bet is undefined before any pressure has been applied."""
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["check"] * 8)

    assert stats.decisions_facing_a_bet == 0
    assert classifier.classify(stats) == "unknown"


def test_aggression_is_detected_before_pressure_is_required():
    """A raise-heavy opponent is identifiable without ever facing a bet."""
    classifier = RuleBasedOpponentClassifier(min_actions=5)
    stats = build_stats(["raise"] * 6 + ["check"] * 2)

    assert stats.decisions_facing_a_bet == 0
    assert classifier.classify(stats) == "aggressive"


def test_free_checks_do_not_make_a_tight_opponent_look_passive():
    tight_under_pressure = build_stats(["fold"] * 8 + ["call", "call"])
    same_player_left_alone = build_stats(
        ["check"] * 40 + ["fold"] * 8 + ["call", "call"]
    )

    classifier = RuleBasedOpponentClassifier(min_actions=5)

    assert classifier.classify(tight_under_pressure) == "tight"
    assert classifier.classify(same_player_left_alone) == "tight"
