import random

from src.players.base.player_template import PlayerTemplate
from src.poker.betting import call_cost
from src.poker.round_state_utils import get_player_stack


class CallingPlayer(PlayerTemplate):
    """
    Simple stochastic calling-station training opponent.

    Free checks are always taken. With the default probabilities, paid
    decisions are 90% calls, 8% folds, and 2% minimum raises. The player
    intentionally ignores hand strength and bet size so that it remains a clear
    call-heavy behavioural profile while being distinct from the deterministic
    AlwaysCallPlayer evaluation baseline.
    """

    def __init__(
        self,
        player_name: str = "calling_player",
        rng: random.Random | None = None,
        call_probability: float = 0.90,
        raise_probability: float = 0.02,
    ):
        super().__init__(player_name=player_name)
        self._validate_probabilities(
            call_probability=call_probability,
            raise_probability=raise_probability,
        )
        self.rng = rng if rng is not None else random
        self.call_probability = call_probability
        self.raise_probability = raise_probability
        self.reset_tracking()

    def declare_action(self, valid_actions, hole_card, round_state):
        if not valid_actions:
            return "fold", 0

        action_by_name = {
            action["action"]: action
            for action in valid_actions
        }

        call_action = action_by_name.get("call")
        fold_action = action_by_name.get("fold")
        raise_action = action_by_name.get("raise")

        if call_action is None:
            if fold_action is not None:
                return fold_action["action"], fold_action["amount"]

            first_action = valid_actions[0]
            return first_action["action"], first_action["amount"]

        call_amount = call_cost(valid_actions, round_state, self.player_uuid)
        if call_amount <= 0:
            return call_action["action"], call_action["amount"]

        roll = self.rng.random()

        if (
            raise_action is not None
            and self._is_valid_raise(raise_action)
            and roll < self.raise_probability
        ):
            return "raise", self._minimum_raise_amount(raise_action)

        if roll < self.raise_probability + self.call_probability:
            return call_action["action"], call_action["amount"]

        if fold_action is not None:
            return fold_action["action"], fold_action["amount"]

        return call_action["action"], call_action["amount"]

    def receive_game_start_message(self, game_info):
        super().receive_game_start_message(game_info)

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        current_stack = get_player_stack(round_state, self.uuid)
        self.update_tracking_after_round(current_stack=current_stack)

    @staticmethod
    def _validate_probabilities(
        *,
        call_probability: float,
        raise_probability: float,
    ) -> None:
        for name, probability in (
            ("call_probability", call_probability),
            ("raise_probability", raise_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{name} must be in range [0, 1]"
                )

        if call_probability + raise_probability > 1.0:
            raise ValueError(
                "call_probability + raise_probability must not exceed 1"
            )

    @staticmethod
    def _is_valid_raise(raise_action: dict) -> bool:
        amount = raise_action.get("amount", 0)

        if isinstance(amount, dict):
            minimum = int(amount.get("min", -1))
            maximum = int(amount.get("max", -1))
            return minimum > 0 and maximum >= minimum

        return int(amount) > 0

    @staticmethod
    def _minimum_raise_amount(raise_action: dict) -> int:
        amount = raise_action.get("amount", 0)

        if isinstance(amount, dict):
            return int(amount["min"])

        return int(amount)
