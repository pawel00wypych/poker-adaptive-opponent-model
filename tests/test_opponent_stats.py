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