from src.players.constants import PLAYER_NAME_ALWAYS_RAISE
from src.players.base.player_template import PlayerTemplate


class AlwaysRaisePlayer(PlayerTemplate):
    """
    Deterministic sanity-check baseline.

    Strategy:
    - raise the minimum legal amount whenever raise is available,
    - otherwise call/check,
    - otherwise fold.

    This player does not use cards, opponent modelling or learning. It is
    intended to detect whether scripted opponents can be exploited by a very
    simple aggressive policy.
    """

    def __init__(
        self,
        player_name: str = PLAYER_NAME_ALWAYS_RAISE,
    ):
        super().__init__(player_name=player_name)

    def declare_action(
        self,
        valid_actions,
        hole_card,
        round_state,
    ):
        raise_action = self._find_action(
            valid_actions,
            "raise",
        )

        if raise_action is not None:
            amount = self._extract_min_raise_amount(
                raise_action,
            )

            if amount is not None:
                return "raise", amount

        call_action = self._find_action(
            valid_actions,
            "call",
        )

        if call_action is not None:
            return "call", call_action["amount"]

        fold_action = self._find_action(
            valid_actions,
            "fold",
        )

        if fold_action is not None:
            return "fold", fold_action["amount"]

        first = valid_actions[0]
        return first["action"], first["amount"]

    def receive_game_update_message(
        self,
        action,
        round_state,
    ):
        # Tracking stack after each intermediate action is unnecessary for
        # this deterministic baseline. The final stack is handled at round end.
        pass

    def receive_round_result_message(
        self,
        winners,
        hand_info,
        round_state,
    ):
        final_stack = self.get_my_stack_from_round_state(
            round_state,
        )

        self.update_round_tracking_after_result(
            final_stack,
        )

    @staticmethod
    def _find_action(
        valid_actions,
        action_name: str,
    ):
        return next(
            (
                action
                for action in valid_actions
                if action["action"] == action_name
            ),
            None,
        )

    @staticmethod
    def _extract_min_raise_amount(
        raise_action,
    ):
        amount = raise_action["amount"]

        if isinstance(amount, dict):
            min_raise = amount.get("min")
            max_raise = amount.get("max")

            if (
                min_raise is None
                or max_raise is None
                or min_raise == -1
                or max_raise == -1
            ):
                return None

            return min_raise

        return amount
