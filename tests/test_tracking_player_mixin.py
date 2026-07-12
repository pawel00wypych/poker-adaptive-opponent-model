from src.agents.tracking_player_mixin import TrackingPlayerMixin


class DummyTrackedPlayer(TrackingPlayerMixin):
    def __init__(self):
        self.reset_tracking()


def test_tracking_initial_state():
    player = DummyTrackedPlayer()

    assert player.hands_played == 0
    assert player.total_reward_bb == 0.0
    assert player.previous_stack is None
    assert player.initial_stack is None


def test_tracking_first_round_has_zero_reward():
    player = DummyTrackedPlayer()

    reward_bb = player.update_tracking_after_round(current_stack=100, big_blind=10)

    assert reward_bb == 0.0
    assert player.total_reward_bb == 0.0
    assert player.hands_played == 1
    assert player.initial_stack == 100
    assert player.previous_stack == 100


def test_tracking_positive_reward():
    player = DummyTrackedPlayer()

    player.update_tracking_after_round(current_stack=100, big_blind=10)
    reward_bb = player.update_tracking_after_round(current_stack=130, big_blind=10)

    assert reward_bb == 3.0
    assert player.total_reward_bb == 3.0
    assert player.hands_played == 2
    assert player.previous_stack == 130


def test_tracking_negative_reward():
    player = DummyTrackedPlayer()

    player.update_tracking_after_round(current_stack=100, big_blind=10)
    reward_bb = player.update_tracking_after_round(current_stack=70, big_blind=10)

    assert reward_bb == -3.0
    assert player.total_reward_bb == -3.0
    assert player.hands_played == 2
    assert player.previous_stack == 70