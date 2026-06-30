from typing import Any


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
    def to_engine_action(action_id: int, valid_actions: list[dict[str, Any]]) -> tuple[str, int]:
        action_by_name = {item["action"]: item for item in valid_actions}

        if action_id == ActionMapper.FOLD:
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
        available = {item["action"] for item in valid_actions}
        legal = []

        if "fold" in available:
            legal.append(ActionMapper.FOLD)

        if "call" in available:
            legal.append(ActionMapper.CALL)

        raise_action = next((item for item in valid_actions if item["action"] == "raise"), None)

        if raise_action is not None:
            amount = raise_action["amount"]
            if isinstance(amount, dict):
                min_raise = amount.get("min")
                max_raise = amount.get("max")
                if min_raise is not None and max_raise is not None and min_raise != -1 and max_raise != -1:
                    legal.append(ActionMapper.RAISE_MIN)
            else:
                legal.append(ActionMapper.RAISE_MIN)

        return legal or [ActionMapper.CALL]

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