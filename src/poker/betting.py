"""Translate PyPokerEngine betting amounts into what an action actually costs.

PyPokerEngine reports the ``call`` amount in ``valid_actions`` as the bet
*level* a player must match, not as the chips that leaving the hand would
require. The engine subtracts what the player already committed on the current
street::

    need_amount_for_action(player, amount) = amount - player.paid_sum()

Reading that field as a cost is wrong whenever a player already has chips in
for the street. The clearest case is the big blind before the flop: it faces a
level of 10 having already posted 10, so calling is free even though the field
reads 10.

Decision-making code should work with real costs, while the engine must keep
receiving levels. ``to_decision_actions`` performs that conversion at the
player boundary; the original ``valid_actions`` stay untouched for the reply.
"""

from typing import Any

NON_PAYING_ACTIONS = {"FOLD", "ANTE"}


def paid_this_street(
    round_state: dict[str, Any],
    player_uuid: str,
) -> int:
    """Return the level a player has already committed on the current street.

    Mirrors ``Player.paid_sum``: the amount of the most recent paying action,
    because engine history entries store cumulative levels rather than
    increments.
    """
    street = round_state.get("street")
    histories = round_state.get("action_histories", {}) or {}
    street_histories = histories.get(street) or []

    paying = [
        entry
        for entry in street_histories
        if entry.get("uuid") == player_uuid
        and entry.get("action") not in NON_PAYING_ACTIONS
    ]

    if not paying:
        return 0

    return int(paying[-1].get("amount", 0))


def find_action(
    valid_actions: list[dict[str, Any]],
    action_name: str,
) -> dict[str, Any] | None:
    return next(
        (
            action
            for action in valid_actions
            if action.get("action") == action_name
        ),
        None,
    )


def call_cost(
    valid_actions: list[dict[str, Any]],
    round_state: dict[str, Any],
    player_uuid: str,
) -> int:
    """Return the chips a call actually costs, never below zero."""
    call_action = find_action(valid_actions, "call")

    if call_action is None:
        return 0

    level = int(call_action.get("amount", 0))

    return max(
        0,
        level - paid_this_street(round_state, player_uuid),
    )


def to_decision_actions(
    valid_actions: list[dict[str, Any]],
    round_state: dict[str, Any],
    player_uuid: str,
) -> list[dict[str, Any]]:
    """Return a copy of ``valid_actions`` whose call amount is the real cost.

    Raise amounts are left alone: they are levels for both the engine and for
    the minimum-raise action the agents use.

    The result is for decisions only. Replies to the engine must be built from
    the original ``valid_actions``, otherwise the engine would subtract the
    already-paid amount a second time.
    """
    cost = call_cost(valid_actions, round_state, player_uuid)

    decision_actions = []
    for action in valid_actions:
        if action.get("action") == "call":
            decision_actions.append({**action, "amount": cost})
        else:
            decision_actions.append(dict(action))

    return decision_actions


def is_free_check(
    valid_actions: list[dict[str, Any]],
    round_state: dict[str, Any],
    player_uuid: str,
) -> bool:
    """Report whether staying in the hand costs nothing."""
    return (
        find_action(valid_actions, "call") is not None
        and call_cost(valid_actions, round_state, player_uuid) == 0
    )
