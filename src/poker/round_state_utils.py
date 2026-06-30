from typing import Any


def get_player_seat(round_state: dict[str, Any], player_uuid: str) -> dict[str, Any] | None:
    """
    Returns player's seat information from PyPokerEngine round_state.

    Expected structure:
    round_state["seats"] = [
        {"name": "...", "uuid": "...", "stack": 100, "state": "participating"},
        ...
    ]
    """
    seats = round_state.get("seats", [])

    for seat in seats:
        if seat.get("uuid") == player_uuid:
            return seat

    return None


def get_player_stack(round_state: dict[str, Any], player_uuid: str) -> int:
    seat = get_player_seat(round_state, player_uuid)

    if seat is None:
        return 0

    return int(seat.get("stack", 0))


def get_player_name(round_state: dict[str, Any], player_uuid: str) -> str | None:
    seat = get_player_seat(round_state, player_uuid)

    if seat is None:
        return None

    return seat.get("name")


def get_alive_players(round_state: dict[str, Any]) -> list[dict[str, Any]]:
    seats = round_state.get("seats", [])

    return [
        seat
        for seat in seats
        if int(seat.get("stack", 0)) > 0
    ]


def get_round_count(round_state: dict[str, Any]) -> int:
    """
    PyPokerEngine usually exposes round_count in round_state.
    Fallback to 0 if missing.
    """
    return int(round_state.get("round_count", 0))


def get_main_pot_amount(round_state: dict[str, Any]) -> int:
    return int(
        round_state
        .get("pot", {})
        .get("main", {})
        .get("amount", 0)
    )