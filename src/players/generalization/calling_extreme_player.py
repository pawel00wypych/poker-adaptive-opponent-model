import random

from src.features.hand_strength_encoder import HandStrengthEncoder
from src.players.base.player_template import PlayerTemplate
from src.poker.betting import call_cost
from src.poker.round_state_utils import get_player_stack


WEAK_HAND_STRENGTH_BIN = 0
MEDIUM_HAND_STRENGTH_BIN = 2
STRONG_HAND_STRENGTH_BIN = 3


class CallingExtremePlayer(PlayerTemplate):
    """
    More selective calling-family opponent for generalization tests.

    The player is still mostly passive and call-heavy, but it is less willing to
    pay expensive calls with weak hands and can occasionally value-raise with a
    strong hand. It is intentionally weaker and simpler than RuleBasedPlayer.
    """

    def __init__(
        self,
        player_name: str = "calling_extreme",
        rng: random.Random | None = None,
        call_probability: float = 0.82,
        max_call_stack_ratio: float = 0.40,
        weak_expensive_fold_probability: float = 0.60,
        medium_expensive_fold_probability: float = 0.35,
        strong_expensive_fold_probability: float = 0.10,
        strong_raise_probability: float = 0.08,
    ):
        super().__init__(player_name=player_name)
        self.rng = rng if rng is not None else random
        self.call_probability = call_probability
        self.max_call_stack_ratio = max_call_stack_ratio
        self.weak_expensive_fold_probability = weak_expensive_fold_probability
        self.medium_expensive_fold_probability = medium_expensive_fold_probability
        self.strong_expensive_fold_probability = strong_expensive_fold_probability
        self.strong_raise_probability = strong_raise_probability
        self.reset_tracking()

    def declare_action(self, valid_actions, hole_card, round_state):
        if not valid_actions:
            return "fold", 0

        action_by_name = {
            action["action"]: action
            for action in valid_actions
        }

        call_action = action_by_name.get("call")
        raise_action = action_by_name.get("raise")
        fold_action = action_by_name.get("fold")
        hand_strength = self._hand_strength_bin(hole_card, round_state)

        if (
            raise_action is not None
            and self._is_valid_raise(raise_action)
            and hand_strength >= STRONG_HAND_STRENGTH_BIN
            and self.rng.random() < self.strong_raise_probability
        ):
            return "raise", self._minimum_raise_amount(raise_action)

        if call_action is not None:
            call_amount = call_cost(valid_actions, round_state, self.player_uuid)

            if call_amount <= 0:
                return call_action["action"], call_action["amount"]

            if (
                fold_action is not None
                and self._is_expensive_call(call_amount, round_state)
                and self.rng.random()
                < self._fold_expensive_probability(hand_strength)
            ):
                return fold_action["action"], fold_action["amount"]

            if self.rng.random() < self.call_probability:
                return call_action["action"], call_action["amount"]

        if fold_action is not None:
            return fold_action["action"], fold_action["amount"]

        if call_action is not None:
            return call_action["action"], call_action["amount"]

        first_action = valid_actions[0]
        return first_action["action"], first_action["amount"]

    def _is_expensive_call(self, call_amount: int, round_state: dict) -> bool:
        """Judge cost against the stack, using what calling actually costs."""
        if call_amount <= 0:
            return False

        stack = get_player_stack(round_state, self.uuid)
        if stack <= 0:
            return False

        return call_amount / stack > self.max_call_stack_ratio

    def _fold_expensive_probability(self, hand_strength: int) -> float:
        if hand_strength >= STRONG_HAND_STRENGTH_BIN:
            return self.strong_expensive_fold_probability

        if hand_strength <= WEAK_HAND_STRENGTH_BIN:
            return self.weak_expensive_fold_probability

        if hand_strength <= MEDIUM_HAND_STRENGTH_BIN:
            return self.medium_expensive_fold_probability

        return self.medium_expensive_fold_probability

    @staticmethod
    def _hand_strength_bin(hole_card: list[str], round_state: dict) -> int:
        if len(hole_card) != 2:
            return MEDIUM_HAND_STRENGTH_BIN

        try:
            return HandStrengthEncoder.encode(
                hole_cards=hole_card,
                community_cards=round_state.get("community_card", []),
            )
        except ValueError:
            return WEAK_HAND_STRENGTH_BIN

    @staticmethod
    def _is_valid_raise(raise_action: dict) -> bool:
        amount = raise_action.get("amount", 0)

        if isinstance(amount, dict):
            return (
                int(amount.get("min", -1)) > 0
                and int(amount.get("max", -1)) >= int(amount.get("min", -1))
            )

        return int(amount) > 0

    @staticmethod
    def _minimum_raise_amount(raise_action: dict) -> int:
        amount = raise_action.get("amount", 0)

        if isinstance(amount, dict):
            return int(amount["min"])

        return int(amount)
