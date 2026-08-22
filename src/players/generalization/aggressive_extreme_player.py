import random

from src.features.hand_strength_encoder import HandStrengthEncoder
from src.players.base.player_template import PlayerTemplate
from src.poker.betting import avoid_free_fold

WEAK_HAND_STRENGTH_BIN = 0
STRONG_HAND_STRENGTH_BIN = 3


class AggressiveExtremePlayer(PlayerTemplate):
    """
    Stronger aggressive-family opponent for generalization tests.

    The player is intentionally very aggressive, but it is not equivalent to
    AlwaysRaisePlayer because it can still call or fold with some probability.
    """

    def __init__(
        self,
        player_name: str = "aggressive_extreme",
        rng: random.Random | None = None,
    ):
        super().__init__(player_name=player_name)
        self.rng = rng if rng is not None else random

    def declare_action(self, valid_actions, hole_card, round_state):
        action, amount = _choose_aggressive_action(
            valid_actions=valid_actions,
            hole_card=hole_card,
            round_state=round_state,
            rng=self.rng,
            base_raise_probability_by_street={
                "preflop": 0.78,
                "flop": 0.84,
                "turn": 0.84,
                "river": 0.80,
            },
            call_probability=0.14,
            strong_raise_bonus=0.10,
            weak_raise_penalty=0.22,
            max_raise_probability=0.18,
        )

        return avoid_free_fold(
            action,
            amount,
            valid_actions,
            round_state,
            self.player_uuid,
        )


def _choose_aggressive_action(
    *,
    valid_actions,
    hole_card,
    round_state,
    rng: random.Random,
    base_raise_probability_by_street: dict[str, float],
    call_probability: float,
    strong_raise_bonus: float,
    weak_raise_penalty: float,
    max_raise_probability: float,
):
    if not valid_actions:
        return "fold", 0

    action_by_name = {
        action["action"]: action
        for action in valid_actions
    }

    raise_action = action_by_name.get("raise")
    call_action = action_by_name.get("call")
    fold_action = action_by_name.get("fold")

    street = round_state.get("street", "preflop")
    raise_probability = base_raise_probability_by_street.get(
        street,
        base_raise_probability_by_street["preflop"],
    )

    hand_strength = _hand_strength_bin(hole_card, round_state)
    if hand_strength >= STRONG_HAND_STRENGTH_BIN:
        raise_probability += strong_raise_bonus
    elif hand_strength <= WEAK_HAND_STRENGTH_BIN:
        raise_probability -= weak_raise_penalty

    raise_probability = _clamp_probability(raise_probability)
    roll = rng.random()

    if (
        raise_action is not None
        and _is_valid_raise(raise_action)
        and roll < raise_probability
    ):
        return "raise", _select_raise_amount(
            raise_action=raise_action,
            rng=rng,
            max_raise_probability=max_raise_probability,
        )

    if call_action is not None and roll < raise_probability + call_probability:
        return call_action["action"], call_action["amount"]

    if fold_action is not None:
        return fold_action["action"], fold_action["amount"]

    if call_action is not None:
        return call_action["action"], call_action["amount"]

    first_action = valid_actions[0]
    return first_action["action"], first_action["amount"]


def _hand_strength_bin(hole_card: list[str], round_state: dict) -> int:
    if len(hole_card) != 2:
        return WEAK_HAND_STRENGTH_BIN + 1

    try:
        return HandStrengthEncoder.encode(
            hole_cards=hole_card,
            community_cards=round_state.get("community_card", []),
        )
    except ValueError:
        return WEAK_HAND_STRENGTH_BIN


def _is_valid_raise(raise_action: dict) -> bool:
    amount = raise_action.get("amount", 0)

    if isinstance(amount, dict):
        return (
            int(amount.get("min", -1)) > 0
            and int(amount.get("max", -1)) >= int(amount.get("min", -1))
        )

    return int(amount) > 0


def _select_raise_amount(
    *,
    raise_action: dict,
    rng: random.Random,
    max_raise_probability: float,
) -> int:
    amount = raise_action.get("amount", 0)

    if isinstance(amount, dict):
        if rng.random() < max_raise_probability and int(amount.get("max", -1)) > 0:
            return int(amount["max"])

        return int(amount["min"])

    return int(amount)


def _clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, value))
