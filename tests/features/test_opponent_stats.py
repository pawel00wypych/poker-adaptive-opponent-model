from src.features.opponent_stats import OpponentStats


def test_empty_stats_have_zero_rates():
    stats = OpponentStats()

    assert stats.total_actions == 0
    assert stats.fold_rate == 0.0
    assert stats.call_rate == 0.0
    assert stats.raise_rate == 0.0
    assert stats.aggression_ratio == 0.0


def test_update_action_counts_fold_call_raise():
    stats = OpponentStats()

    stats.update_action("fold")
    stats.update_action("call")
    stats.update_action("raise")
    stats.update_action("raise")

    assert stats.folds == 1
    assert stats.calls == 1
    assert stats.raises == 2
    assert stats.total_actions == 4


def test_rates_are_calculated_correctly():
    stats = OpponentStats()

    stats.update_action("fold")
    stats.update_action("call")
    stats.update_action("raise")
    stats.update_action("raise")

    assert stats.fold_rate == 0.25
    assert stats.call_rate == 0.25
    assert stats.raise_rate == 0.5


def test_aggression_ratio():
    stats = OpponentStats()

    stats.update_action("call")
    stats.update_action("fold")
    stats.update_action("raise")
    stats.update_action("raise")

    assert stats.aggression_ratio == 1.0


def test_finish_hand_increments_hands_observed():
    stats = OpponentStats()

    stats.finish_hand()
    stats.finish_hand()

    assert stats.hands_observed == 2


def test_unknown_action_is_ignored():
    stats = OpponentStats()

    stats.update_action("small_blind")
    stats.update_action("big_blind")
    stats.update_action("unknown")

    assert stats.total_actions == 0

def test_check_is_recorded_separately_from_a_paid_call():
    stats = OpponentStats()

    stats.update_action("call", paid=0)
    stats.update_action("call", paid=10)

    assert stats.checks == 1
    assert stats.calls == 1
    assert stats.total_actions == 2


def test_unknown_increment_is_treated_as_a_paid_call():
    """Conservative default: never invent a free check."""
    stats = OpponentStats()

    stats.update_action("call")

    assert stats.calls == 1
    assert stats.checks == 0


def test_decisions_facing_a_bet_excludes_checks_and_raises():
    stats = OpponentStats()
    stats.update_action("call", paid=0)
    stats.update_action("call", paid=10)
    stats.update_action("fold")
    stats.update_action("raise")

    assert stats.decisions_facing_a_bet == 2


def test_fold_to_bet_rate_ignores_free_checks():
    stats = OpponentStats()
    for _ in range(50):
        stats.update_action("call", paid=0)
    for _ in range(3):
        stats.update_action("fold")
    stats.update_action("call", paid=10)

    assert stats.fold_rate < 0.06
    assert stats.fold_to_bet_rate == 0.75


def test_fold_to_bet_rate_is_zero_without_any_bet_decisions():
    stats = OpponentStats()
    stats.update_action("call", paid=0)

    assert stats.decisions_facing_a_bet == 0
    assert stats.fold_to_bet_rate == 0.0


def test_as_dict_exposes_the_new_counters():
    stats = OpponentStats()
    stats.update_action("call", paid=0)
    stats.update_action("fold")

    payload = stats.as_dict()

    assert payload["checks"] == 1
    assert payload["folds"] == 1
    assert "fold_to_bet_rate" in payload
    assert "decisions_facing_a_bet" in payload
