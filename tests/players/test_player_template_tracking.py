from src.players.base.player_template import PlayerTemplate


class DummyTrackedPlayer(PlayerTemplate):
    def __init__(self):
        super().__init__(player_name="dummy_tracked")
        self.reset_tracking()


def test_tracking_initial_state():
    player = DummyTrackedPlayer()

    assert player.hands_played == 0
    assert player.total_reward_bb == 0.0
    assert player.hand_start_stack is None
    assert player.initial_stack is None


def test_tracking_first_round_has_zero_reward():
    player = DummyTrackedPlayer()

    reward_bb = player.update_tracking_after_round(current_stack=100)

    assert reward_bb == 0.0
    assert player.total_reward_bb == 0.0
    assert player.hands_played == 1
    assert player.initial_stack == 100
    assert player.hand_start_stack == 100


def test_tracking_positive_reward():
    player = DummyTrackedPlayer()

    player.update_tracking_after_round(current_stack=100)
    reward_bb = player.update_tracking_after_round(current_stack=130)

    assert reward_bb == 3.0
    assert player.total_reward_bb == 3.0
    assert player.hands_played == 2
    assert player.hand_start_stack == 130


def test_tracking_negative_reward():
    player = DummyTrackedPlayer()

    player.update_tracking_after_round(current_stack=100)
    reward_bb = player.update_tracking_after_round(current_stack=70)

    assert reward_bb == -3.0
    assert player.total_reward_bb == -3.0
    assert player.hands_played == 2
    assert player.hand_start_stack == 70