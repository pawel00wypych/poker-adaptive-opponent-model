from src.players.base.player_template import PlayerTemplate
from src.poker.betting import call_cost
from src.poker.round_state_utils import get_player_stack

FOLD_THRESHOLD_IN_BIG_BLINDS = 3
RAISE_EVERY_NTH_DECISION = 10


class RuleBasedPlayer(PlayerTemplate):
    """
    Simple rule-based baseline.

    Strategy:
    - check/call if free,
    - fold calls costing three big blinds or more,
    - min-raise on every tenth decision,
    - otherwise call cheap actions.

    This player does not use cards, opponent modelling or learning.

    It must behave **identically in every game**, because it is the reference
    point for ``delta_vs_rule_based``. That requires the raise cadence to
    restart at each game start; otherwise the baseline's behaviour depends on
    how many hands it happened to play earlier, and the comparison measures the
    order games were run in as much as it measures the agent.

    The fold threshold is expressed in big blinds rather than chips so that it
    keeps its meaning if the blind structure changes. At the default
    ``small_blind_amount=5`` it evaluates to 30 chips, which is exactly the
    constant it replaces.
    """

    def __init__(self, player_name: str = "rule_based"):
        super().__init__(player_name=player_name)
        self.action_counter = 0
        self.reset_tracking()

    @property
    def fold_threshold(self) -> int:
        return FOLD_THRESHOLD_IN_BIG_BLINDS * self.big_blind_amount

    def declare_action(self, valid_actions, hole_card, round_state):
        self.action_counter += 1

        fold_action = self._find_action(valid_actions, "fold")
        call_action = self._find_action(valid_actions, "call")
        raise_action = self._find_action(valid_actions, "raise")

        call_amount = (
            call_cost(valid_actions, round_state, self.player_uuid)
            if call_action
            else 0
        )

        if call_action and call_amount == 0:
            # Decisions use the real cost, but the engine reply needs the
            # original bet level even when matching it costs nothing.
            return call_action["action"], call_action["amount"]

        if call_amount >= self.fold_threshold and fold_action:
            return "fold", fold_action["amount"]

        if self.action_counter % RAISE_EVERY_NTH_DECISION == 0 and (
            raise_action is not None
        ):
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
                    return "raise", min_raise

        if call_action:
            return "call", call_action["amount"]

        if fold_action:
            return "fold", fold_action["amount"]

        first = valid_actions[0]
        return first["action"], first["amount"]

    def receive_game_start_message(self, game_info):
        super().receive_game_start_message(game_info)
        self.action_counter = 0

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        current_stack = get_player_stack(round_state, self.uuid)
        self.update_tracking_after_round(current_stack=current_stack)

    @staticmethod
    def _find_action(valid_actions, action_name):
        return next(
            (item for item in valid_actions if item["action"] == action_name),
            None,
        )
