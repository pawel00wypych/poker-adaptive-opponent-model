class TrackingPlayerMixin:
    """
    Adds basic tracking for evaluation.

    This mixin assumes PyPokerEngine calls receive_game_start_message
    and receive_round_result_message on the player.
    """

    def __init__(self):
        self.initial_stack = None
        self.previous_stack = None
        self.total_reward_bb = 0.0
        self.hands_played = 0

    def reset_tracking(self) -> None:
        self.hands_played = 0
        self.total_reward_bb = 0.0
        self.previous_stack = None
        self.initial_stack = None

    def update_tracking_after_round(self, current_stack: int, big_blind: int = 10) -> float:
        if self.initial_stack is None:
            self.initial_stack = current_stack

        if self.previous_stack is None:
            self.previous_stack = current_stack

        reward = current_stack - self.previous_stack
        reward_bb = reward / big_blind

        self.total_reward_bb += reward_bb
        self.previous_stack = current_stack
        self.hands_played += 1

        return reward_bb