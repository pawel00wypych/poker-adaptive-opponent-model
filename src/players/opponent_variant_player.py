import random
from dataclasses import dataclass
from typing import Mapping

from src.features.hand_strength_encoder import HandStrengthEncoder
from src.players.constants import (
    AGGRESSIVE_OPPONENT_VARIANTS,
    CALLING_OPPONENT_VARIANTS,
    GENERALIZATION_OPPONENT_VARIANTS,
    OPPONENT_VARIANT_AGGRESSIVE_EXTREME,
    OPPONENT_VARIANT_AGGRESSIVE_LIGHT,
    OPPONENT_VARIANT_CALLING_MEDIUM,
    OPPONENT_VARIANT_CALLING_STRONG,
    OPPONENT_VARIANT_CALLING_WEAK,
)
from src.players.player_template import PlayerTemplate
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
)
from src.poker.round_state_utils import get_player_stack


OPPONENT_VARIANT_TO_BASE_TYPE = {
    OPPONENT_VARIANT_CALLING_WEAK: OPPONENT_TYPE_CALLING,
    OPPONENT_VARIANT_CALLING_MEDIUM: OPPONENT_TYPE_CALLING,
    OPPONENT_VARIANT_CALLING_STRONG: OPPONENT_TYPE_CALLING,
    OPPONENT_VARIANT_AGGRESSIVE_LIGHT: OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_VARIANT_AGGRESSIVE_EXTREME: OPPONENT_TYPE_AGGRESSIVE,
}

WEAK_HAND_STRENGTH_BIN = 0
STRONG_HAND_STRENGTH_BIN = 3


@dataclass(frozen=True)
class CallingVariantConfig:
    """
    Behaviour parameters for passive opponent variants used in generalization tests.

    Variants are intentionally not part of the training opponent distribution.
    They approximate the same broad family as the base CallingPlayer, but with
    different willingness to call expensive bets and a small optional raise rate.
    """

    name: str
    call_probability: float
    raise_probability: float
    max_call_stack_ratio: float
    fold_expensive_probability: float
    hand_strength_raise_bonus: float = 0.0
    weak_hand_raise_penalty: float = 0.0


@dataclass(frozen=True)
class AggressiveVariantConfig:
    """
    Behaviour parameters for aggressive opponent variants used in generalization tests.

    The extreme variant is deliberately not identical to AlwaysRaisePlayer. It
    still calls or folds with small probability, so it remains a behavioural
    variant rather than a deterministic sanity baseline.
    """

    name: str
    raise_probability_by_street: Mapping[str, float]
    call_probability: float
    raise_max_probability: float = 0.0
    hand_strength_raise_bonus: float = 0.0
    weak_hand_raise_penalty: float = 0.0


CALLING_VARIANT_CONFIGS = {
    OPPONENT_VARIANT_CALLING_WEAK: CallingVariantConfig(
        name=OPPONENT_VARIANT_CALLING_WEAK,
        call_probability=0.98,
        raise_probability=0.00,
        max_call_stack_ratio=1.00,
        fold_expensive_probability=0.00,
        hand_strength_raise_bonus=0.00,
        weak_hand_raise_penalty=0.00,
    ),
    OPPONENT_VARIANT_CALLING_MEDIUM: CallingVariantConfig(
        name=OPPONENT_VARIANT_CALLING_MEDIUM,
        call_probability=0.88,
        raise_probability=0.02,
        max_call_stack_ratio=0.50,
        fold_expensive_probability=0.25,
        hand_strength_raise_bonus=0.08,
        weak_hand_raise_penalty=0.01,
    ),
    OPPONENT_VARIANT_CALLING_STRONG: CallingVariantConfig(
        name=OPPONENT_VARIANT_CALLING_STRONG,
        call_probability=0.72,
        raise_probability=0.08,
        max_call_stack_ratio=0.30,
        fold_expensive_probability=0.65,
        hand_strength_raise_bonus=0.15,
        weak_hand_raise_penalty=0.04,
    ),
}

AGGRESSIVE_VARIANT_CONFIGS = {
    OPPONENT_VARIANT_AGGRESSIVE_LIGHT: AggressiveVariantConfig(
        name=OPPONENT_VARIANT_AGGRESSIVE_LIGHT,
        raise_probability_by_street={
            "preflop": 0.45,
            "flop": 0.55,
            "turn": 0.55,
            "river": 0.50,
        },
        call_probability=0.35,
        raise_max_probability=0.00,
        hand_strength_raise_bonus=0.20,
        weak_hand_raise_penalty=0.20,
    ),
    OPPONENT_VARIANT_AGGRESSIVE_EXTREME: AggressiveVariantConfig(
        name=OPPONENT_VARIANT_AGGRESSIVE_EXTREME,
        raise_probability_by_street={
            "preflop": 0.85,
            "flop": 0.90,
            "turn": 0.90,
            "river": 0.85,
        },
        call_probability=0.12,
        raise_max_probability=0.20,
        hand_strength_raise_bonus=0.10,
        weak_hand_raise_penalty=0.20,
    ),
}


class CallingVariantPlayer(PlayerTemplate):
    """
    Parametrized calling-family opponent for generalization evaluation.
    """

    def __init__(
        self,
        config: CallingVariantConfig,
        rng: random.Random | None = None,
        player_name: str | None = None,
    ):
        super().__init__(
            player_name=player_name or config.name,
        )
        self.config = config
        self.rng = rng or random.Random()

    def declare_action(self, valid_actions, hole_card, round_state):
        if not valid_actions:
            return "fold", 0

        action_by_name = _action_by_name(valid_actions)
        call_action = action_by_name.get("call")
        raise_action = action_by_name.get("raise")
        fold_action = action_by_name.get("fold")

        raise_probability = _adjust_probability_for_hand_strength(
            base_probability=self.config.raise_probability,
            hole_card=hole_card,
            round_state=round_state,
            bonus=self.config.hand_strength_raise_bonus,
            penalty=self.config.weak_hand_raise_penalty,
        )

        if (
            raise_action is not None
            and _is_valid_raise(raise_action)
            and self.rng.random() < raise_probability
        ):
            return "raise", _minimum_raise_amount(raise_action)

        if (
            call_action is not None
            and fold_action is not None
            and self._is_expensive_call(call_action, round_state)
            and self.rng.random() < self.config.fold_expensive_probability
        ):
            return fold_action["action"], fold_action["amount"]

        if (
            call_action is not None
            and self.rng.random() < self.config.call_probability
        ):
            return call_action["action"], call_action["amount"]

        if fold_action is not None:
            return fold_action["action"], fold_action["amount"]

        if call_action is not None:
            return call_action["action"], call_action["amount"]

        return _first_action(valid_actions)

    def _is_expensive_call(self, call_action: dict, round_state: dict) -> bool:
        call_amount = int(call_action.get("amount", 0))

        if call_amount <= 0:
            return False

        stack = get_player_stack(
            round_state,
            self.uuid,
        )

        if stack <= 0:
            return False

        return (
            call_amount / stack
        ) > self.config.max_call_stack_ratio


class AggressiveVariantPlayer(PlayerTemplate):
    """
    Parametrized aggressive-family opponent for generalization evaluation.
    """

    def __init__(
        self,
        config: AggressiveVariantConfig,
        rng: random.Random | None = None,
        player_name: str | None = None,
    ):
        super().__init__(
            player_name=player_name or config.name,
        )
        self.config = config
        self.rng = rng or random.Random()

    def declare_action(self, valid_actions, hole_card, round_state):
        if not valid_actions:
            return "fold", 0

        action_by_name = _action_by_name(valid_actions)
        raise_action = action_by_name.get("raise")
        call_action = action_by_name.get("call")
        fold_action = action_by_name.get("fold")

        raise_probability = self._raise_probability(
            hole_card=hole_card,
            round_state=round_state,
        )

        roll = self.rng.random()

        if (
            raise_action is not None
            and _is_valid_raise(raise_action)
            and roll < raise_probability
        ):
            return "raise", self._select_raise_amount(
                raise_action
            )

        if (
            call_action is not None
            and roll < raise_probability + self.config.call_probability
        ):
            return call_action["action"], call_action["amount"]

        if fold_action is not None:
            return fold_action["action"], fold_action["amount"]

        if call_action is not None:
            return call_action["action"], call_action["amount"]

        return _first_action(valid_actions)

    def _raise_probability(
        self,
        hole_card: list[str],
        round_state: dict,
    ) -> float:
        street = round_state.get("street", "preflop")
        base_probability = self.config.raise_probability_by_street.get(
            street,
            self.config.raise_probability_by_street["preflop"],
        )

        return _adjust_probability_for_hand_strength(
            base_probability=base_probability,
            hole_card=hole_card,
            round_state=round_state,
            bonus=self.config.hand_strength_raise_bonus,
            penalty=self.config.weak_hand_raise_penalty,
        )

    def _select_raise_amount(self, raise_action: dict) -> int:
        amount = raise_action.get("amount", 0)

        if isinstance(amount, dict):
            if (
                self.rng.random() < self.config.raise_max_probability
                and int(amount.get("max", -1)) > 0
            ):
                return int(amount["max"])

            return int(amount["min"])

        return int(amount)


def build_calling_variant_player(
    variant_name: str,
    rng: random.Random | None = None,
) -> CallingVariantPlayer:
    if variant_name not in CALLING_VARIANT_CONFIGS:
        raise ValueError(
            f"Unsupported calling variant: {variant_name}. "
            f"Supported variants: {sorted(CALLING_VARIANT_CONFIGS)}"
        )

    return CallingVariantPlayer(
        config=CALLING_VARIANT_CONFIGS[variant_name],
        rng=rng,
        player_name=variant_name,
    )


def build_aggressive_variant_player(
    variant_name: str,
    rng: random.Random | None = None,
) -> AggressiveVariantPlayer:
    if variant_name not in AGGRESSIVE_VARIANT_CONFIGS:
        raise ValueError(
            f"Unsupported aggressive variant: {variant_name}. "
            f"Supported variants: {sorted(AGGRESSIVE_VARIANT_CONFIGS)}"
        )

    return AggressiveVariantPlayer(
        config=AGGRESSIVE_VARIANT_CONFIGS[variant_name],
        rng=rng,
        player_name=variant_name,
    )


def build_opponent_variant(
    variant_name: str,
    rng: random.Random | None = None,
):
    if variant_name in CALLING_VARIANT_CONFIGS:
        return build_calling_variant_player(
            variant_name=variant_name,
            rng=rng,
        )

    if variant_name in AGGRESSIVE_VARIANT_CONFIGS:
        return build_aggressive_variant_player(
            variant_name=variant_name,
            rng=rng,
        )

    raise ValueError(
        f"Unsupported opponent variant: {variant_name}. "
        f"Supported variants: {sorted(GENERALIZATION_OPPONENT_VARIANTS)}"
    )


def get_opponent_variant_base_type(
    variant_name: str,
) -> str:
    if variant_name not in OPPONENT_VARIANT_TO_BASE_TYPE:
        raise ValueError(
            f"Unsupported opponent variant: {variant_name}. "
            f"Supported variants: {sorted(GENERALIZATION_OPPONENT_VARIANTS)}"
        )

    return OPPONENT_VARIANT_TO_BASE_TYPE[variant_name]


def _action_by_name(valid_actions: list[dict]) -> dict[str, dict]:
    return {
        action["action"]: action
        for action in valid_actions
    }


def _is_valid_raise(raise_action: dict) -> bool:
    amount = raise_action.get("amount", 0)

    if isinstance(amount, dict):
        return (
            int(amount.get("min", -1)) > 0
            and int(amount.get("max", -1)) >= int(amount.get("min", -1))
        )

    return int(amount) > 0


def _minimum_raise_amount(raise_action: dict) -> int:
    amount = raise_action.get("amount", 0)

    if isinstance(amount, dict):
        return int(amount["min"])

    return int(amount)


def _first_action(valid_actions: list[dict]):
    first_action = valid_actions[0]
    return first_action["action"], first_action["amount"]


def _adjust_probability_for_hand_strength(
    base_probability: float,
    hole_card: list[str],
    round_state: dict,
    bonus: float,
    penalty: float,
) -> float:
    hand_strength = _hand_strength_bin(
        hole_card=hole_card,
        round_state=round_state,
    )

    if hand_strength >= STRONG_HAND_STRENGTH_BIN:
        return _clamp_probability(base_probability + bonus)

    if hand_strength <= WEAK_HAND_STRENGTH_BIN:
        return _clamp_probability(base_probability - penalty)

    return _clamp_probability(base_probability)


def _hand_strength_bin(
    hole_card: list[str],
    round_state: dict,
) -> int:
    community_cards = round_state.get(
        "community_card",
        [],
    )

    try:
        return HandStrengthEncoder.encode(
            hole_cards=hole_card,
            community_cards=community_cards,
        )
    except ValueError:
        return WEAK_HAND_STRENGTH_BIN


def _clamp_probability(value: float) -> float:
    return min(
        1.0,
        max(
            0.0,
            value,
        ),
    )
