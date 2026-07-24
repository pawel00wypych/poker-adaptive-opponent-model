class TrackingPlayerMixin:
    """
    Provides reusable tracking logic for poker players.

    The mixin intentionally does not define __init__.
    The concrete player class is responsible for calling reset_tracking().
    """

    def reset_tracking(self) -> None:
        self.hands_played = 0
        self.total_reward_bb = 0.0
        self.hand_start_stack = None
        self.initial_stack = None

    def update_tracking_after_round(
        self,
        current_stack: int,
        big_blind: int = 10,
    ) -> float:
        if big_blind <= 0:
            raise ValueError("big_blind must be greater than zero")

        if self.initial_stack is None:
            self.initial_stack = current_stack

        if self.hand_start_stack is None:
            self.hand_start_stack = current_stack

        reward = current_stack - self.hand_start_stack
        reward_bb = reward / big_blind

        self.total_reward_bb += reward_bb
        self.hand_start_stack = current_stack
        self.hands_played += 1

        return reward_bb