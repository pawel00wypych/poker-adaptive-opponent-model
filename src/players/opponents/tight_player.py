import random

from src.features.hand_strength_encoder import HandStrengthEncoder
from src.players.base.player_template import PlayerTemplate
from src.poker.betting import call_cost
from src.poker.round_state_utils import get_player_stack


WEAK_HAND_STRENGTH_BIN = 0
SPECULATIVE_HAND_STRENGTH_BIN = 1
MEDIUM_HAND_STRENGTH_BIN = 2
STRONG_HAND_STRENGTH_BIN = 3
PREMIUM_HAND_STRENGTH_BIN = 4


class TightPlayer(PlayerTemplate):
    """
    Conservative fold-heavy training opponent.

    TightPlayer is designed to be behaviourally distinct from CallingPlayer:
    it folds weak and expensive
    spots frequently, continues mostly with medium-or-better hands, and only
    rarely raises strong hands. It remains a simple scripted opponent rather
    than a strong poker bot.
    """

    def __init__(
        self,
        player_name: str = "tight",
        rng: random.Random | None = None,
        max_call_stack_ratio: float = 0.2,
        strong_raise_probability: float = 0.2,
        premium_raise_probability: float = 0.35,
    ):
        super().__init__(player_name=player_name)
        self.rng = rng if rng is not None else random
        self.max_call_stack_ratio = max_call_stack_ratio
        self.strong_raise_probability = strong_raise_probability
        self.premium_raise_probability = premium_raise_probability

    def declare_action(self, valid_actions, hole_card, round_state):
        if not valid_actions:
            return "fold", 0

        action_by_name = {
            action["action"]: action
            for action in valid_actions
        }

        fold_action = action_by_name.get("fold")
        call_action = action_by_name.get("call")
        raise_action = action_by_name.get("raise")
        hand_strength = self._hand_strength_bin(hole_card, round_state)

        if self._should_raise(
            raise_action=raise_action,
            hand_strength=hand_strength,
        ):
            return "raise", self._minimum_raise_amount(raise_action)

        if call_action is not None:
            call_amount = call_cost(valid_actions, round_state, self.player_uuid)

            if call_amount <= 0:
                return call_action["action"], call_action["amount"]

            if self._is_expensive_call(call_amount, round_state):
                if self.rng.random() < self._fold_expensive_probability(hand_strength):
                    if fold_action is not None:
                        return fold_action["action"], fold_action["amount"]

            if self.rng.random() < self._continue_probability(hand_strength):
                return call_action["action"], call_action["amount"]

        if fold_action is not None:
            return fold_action["action"], fold_action["amount"]

        if call_action is not None:
            return call_action["action"], call_action["amount"]

        first_action = valid_actions[0]
        return first_action["action"], first_action["amount"]

    def _should_raise(self, *, raise_action: dict | None, hand_strength: int) -> bool:
        if raise_action is None or not self._is_valid_raise(raise_action):
            return False

        if hand_strength >= PREMIUM_HAND_STRENGTH_BIN:
            return self.rng.random() < self.premium_raise_probability

        if hand_strength >= STRONG_HAND_STRENGTH_BIN:
            return self.rng.random() < self.strong_raise_probability

        return False

    def _continue_probability(self, hand_strength: int) -> float:
        if hand_strength <= WEAK_HAND_STRENGTH_BIN:
            return 0.08
        if hand_strength <= SPECULATIVE_HAND_STRENGTH_BIN:
            return 0.22
        if hand_strength <= MEDIUM_HAND_STRENGTH_BIN:
            return 0.52
        if hand_strength <= STRONG_HAND_STRENGTH_BIN:
            return 0.76
        return 0.90

    def _fold_expensive_probability(self, hand_strength: int) -> float:
        if hand_strength <= WEAK_HAND_STRENGTH_BIN:
            return 0.92
        if hand_strength <= SPECULATIVE_HAND_STRENGTH_BIN:
            return 0.78
        if hand_strength <= MEDIUM_HAND_STRENGTH_BIN:
            return 0.45
        if hand_strength <= STRONG_HAND_STRENGTH_BIN:
            return 0.18
        return 0.08

    def _is_expensive_call(self, call_amount: int, round_state: dict) -> bool:
        """Judge cost against the stack, using what calling actually costs."""
        if call_amount <= 0:
            return False

        stack = get_player_stack(round_state, self.uuid)
        if stack <= 0:
            return False

        return call_amount / stack > self.max_call_stack_ratio

    @staticmethod
    def _hand_strength_bin(hole_card: list[str], round_state: dict) -> int:
        if len(hole_card) != 2:
            return WEAK_HAND_STRENGTH_BIN

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
