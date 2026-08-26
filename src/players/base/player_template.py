from pypokerengine.players import BasePokerPlayer

from src.config import GameConfig

REWARD_VERSION = "reward_bb_v1"
REWARD_FORMULA = "(stack_after_hand - stack_before_hand) / big_blind"


def _read_small_blind_amount(game_info) -> int | None:
    """Pull the small blind out of the engine's game_info, or None.

    Returning None rather than a default keeps "the engine never told us" and
    "the engine told us the default" distinguishable, so a player that was
    never started cannot silently look like a correctly configured one.
    """
    if not isinstance(game_info, dict):
        return None

    rule = game_info.get("rule")
    if not isinstance(rule, dict):
        return None

    small_blind_amount = rule.get("small_blind_amount")
    if isinstance(small_blind_amount, bool):
        return None
    if not isinstance(small_blind_amount, int) or small_blind_amount <= 0:
        return None

    return small_blind_amount


class PlayerTemplate(BasePokerPlayer):
    """Base poker player with shared round tracking and evaluation utilities."""

    def __init__(
        self,
        player_name="",
        hands_played=0,
        total_profit=0,
        stack=0,
        initial_stack=None,
        total_reward_bb=0.0,
    ):
        super().__init__()

        current_uuid = getattr(self, "uuid", "unknown_uuid")
        self.player_name = str(current_uuid) if player_name == "" else player_name

        self.hands_played = hands_played
        self.total_profit = total_profit
        self.stack = stack
        self.initial_stack = initial_stack
        self.hand_start_stack = None
        self.total_reward_bb = total_reward_bb

        self.hole_card = []
        self.uuid_to_index = None
        self.my_index = None
        self.small_blind_amount = None

    @property
    def big_blind_amount(self):
        """Big blind of the game actually being played.

        The engine reports the blind structure in ``game_info`` at game start,
        so that is what gets used. Reading ``GameConfig.small_blind_amount`` off
        the class instead returns the dataclass *default*, which silently
        decouples every BB-denominated reward from the real blinds - and
        ``mean_profit_bb`` and ``bb_per_100`` are the primary reported metrics.

        Before a game starts - and in unit tests that build a player directly -
        there is nothing to read, so the configured default applies.
        """
        if self.small_blind_amount is None:
            return GameConfig().big_blind_amount

        return self.small_blind_amount * 2

    @property
    def player_uuid(self) -> str | None:
        """Return the engine-assigned uuid, or None before a game starts.

        The engine assigns ``uuid`` when it seats the player, so helpers that
        run before that - and unit tests that build a player directly - must
        not assume the attribute exists.
        """
        return getattr(self, "uuid", None)

    def ensure_player_index(self, seats):
        if self.my_index is not None:
            return

        self.uuid_to_index = {seat["uuid"]: i for i, seat in enumerate(seats)}
        self.my_index = self.uuid_to_index[self.uuid]

    def get_my_stack_from_seats(self, seats):
        self.ensure_player_index(seats)
        return seats[self.my_index]["stack"]

    def get_my_stack_from_round_state(self, round_state):
        seats = round_state["seats"]
        self.ensure_player_index(seats)
        return seats[self.my_index]["stack"]

    def calculate_reward_bb(self, final_stack):
        if self.hand_start_stack is None:
            raise RuntimeError(
                "hand_start_stack is not set. "
                "receive_round_start_message() must be called before "
                "reward calculation."
            )

        reward = final_stack - self.hand_start_stack
        return reward / self.big_blind_amount

    def update_round_tracking_after_result(self, final_stack):
        self.stack = final_stack
        self.hand_start_stack = None
        self.hands_played += 1

    def reset_tracking_stats(self):
        self.hands_played = 0
        self.total_profit = 0
        self.total_reward_bb = 0.0
        self.stack = 0
        self.initial_stack = None
        self.hand_start_stack = None

    def reset_tracking(self) -> None:
        """Reset round/profit tracking shared by simple baseline players."""
        self.reset_tracking_stats()

    def update_tracking_after_round(
        self,
        current_stack: int,
    ) -> float:
        """Update hand-level tracking and return the reward in big blinds.

        This method is intended for simple scripted players that do not need
        custom round-result handling. Learned/adaptive players can still use
        calculate_reward_bb() and update_round_tracking_after_result() when
        they need finer control over logging or reward propagation.
        """

        if self.initial_stack is None:
            self.initial_stack = current_stack

        if self.hand_start_stack is None:
            self.hand_start_stack = current_stack

        reward = current_stack - self.hand_start_stack
        reward_bb = reward / self.big_blind_amount

        self.stack = current_stack
        self.total_reward_bb += reward_bb
        self.hand_start_stack = current_stack
        self.hands_played += 1

        return reward_bb

    def receive_game_start_message(self, game_info):
        self.reset_tracking_stats()
        self.small_blind_amount = _read_small_blind_amount(game_info)

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
        self.uuid_to_index = {seat["uuid"]: i for i, seat in enumerate(seats)}
        self.my_index = self.uuid_to_index[self.uuid]

        current_stack = self.get_my_stack_from_seats(seats)

        if self.initial_stack is None:
            self.initial_stack = current_stack

        self.stack = current_stack
        self.hand_start_stack = current_stack

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self.stack = self.get_my_stack_from_round_state(round_state)

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    @classmethod
    def get_action(cls, valid_actions, name):
        for action in valid_actions:
            if action["action"] == name:
                return action

        raise ValueError(
            f"There is no action={name!r} in valid_actions={valid_actions}"
        )
