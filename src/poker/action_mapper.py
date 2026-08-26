from typing import Any

ACTION_VERSION = "action_v1"
ACTION_NAMES = (
    "fold",
    "call_or_check",
    "minimum_raise",
)
INVALID_RAISE_FALLBACK = "call_or_check"
FREE_FOLD_FALLBACK = "check"


class ActionMapper:
    """
    Maps fixed RL action IDs to PyPokerEngine legal actions.

    RL actions:
    0 -> fold
    1 -> call/check
    2 -> raise minimum
    """

    FOLD = 0
    CALL = 1
    RAISE_MIN = 2

    NUM_ACTIONS = 3

    @staticmethod
    def to_engine_action(
        action_id: int,
        valid_actions: list[dict[str, Any]],
    ) -> tuple[str, int]:
        action_by_name = {item["action"]: item for item in valid_actions}

        if action_id == ActionMapper.FOLD:
            if ActionMapper._is_free_call(valid_actions):
                # Folding for free is strictly dominated. A policy trained
                # before fold masking can still request it, so it is redirected
                # to the free check instead of surrendering the hand.
                return ActionMapper._fallback_call(valid_actions)

            fold = action_by_name.get("fold")
            if fold is not None:
                return "fold", fold["amount"]

            return ActionMapper._fallback_call(valid_actions)

        if action_id == ActionMapper.CALL:
            return ActionMapper._fallback_call(valid_actions)

        if action_id == ActionMapper.RAISE_MIN:
            raise_action = action_by_name.get("raise")

            if raise_action is None:
                return ActionMapper._fallback_call(valid_actions)

            amount = raise_action["amount"]

            if isinstance(amount, dict):
                min_raise = amount.get("min")

                if min_raise is None or min_raise == -1:
                    return ActionMapper._fallback_call(valid_actions)

                return "raise", min_raise

            return "raise", amount

        return ActionMapper._fallback_call(valid_actions)

    @staticmethod
    def get_legal_action_ids(valid_actions: list[dict[str, Any]]) -> list[int]:
        """Return legal action ids for the given actions.

        ``valid_actions`` is expected to carry the real call cost, as produced
        by :func:`src.poker.betting.to_decision_actions`. Fold is omitted when
        staying in the hand is free, because giving up a hand for nothing is
        strictly dominated and would otherwise consume exploration budget.
        """
        available = {item["action"] for item in valid_actions}
        legal = []

        if "fold" in available and not ActionMapper._is_free_call(valid_actions):
            legal.append(ActionMapper.FOLD)

        if "call" in available:
            legal.append(ActionMapper.CALL)

        raise_action = next(
            (item for item in valid_actions if item["action"] == "raise"),
            None,
        )

        if raise_action is not None:
            amount = raise_action["amount"]
            if isinstance(amount, dict):
                min_raise = amount.get("min")
                max_raise = amount.get("max")
                if (
                    min_raise is not None
                    and max_raise is not None
                    and min_raise != -1
                    and max_raise != -1
                ):
                    legal.append(ActionMapper.RAISE_MIN)
            else:
                legal.append(ActionMapper.RAISE_MIN)

        return legal or [ActionMapper.CALL]

    @staticmethod
    def _is_free_call(valid_actions: list[dict[str, Any]]) -> bool:
        for item in valid_actions:
            if item["action"] == "call":
                amount = item["amount"]
                return isinstance(amount, int) and amount <= 0

        return False

    @staticmethod
    def _fallback_call(valid_actions: list[dict[str, Any]]) -> tuple[str, int]:
        for item in valid_actions:
            if item["action"] == "call":
                return "call", item["amount"]

        for item in valid_actions:
            if item["action"] == "fold":
                return "fold", item["amount"]

        first = valid_actions[0]
        return first["action"], first["amount"]
