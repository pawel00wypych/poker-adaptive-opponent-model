from src.agents.player_template import PlayerTemplate


class CallingPlayer(PlayerTemplate):
    """
    Passive baseline that calls/checks whenever possible.
    """

    def __init__(self, player_name: str = "calling_player"):
        super().__init__(player_name=player_name)

    def declare_action(self, valid_actions, hole_card, round_state):
        if not valid_actions:
            return "fold", 0

        for action in valid_actions:
            if action["action"] == "call":
                return "call", action["amount"]

        first = valid_actions[0]
        return first["action"], first["amount"]