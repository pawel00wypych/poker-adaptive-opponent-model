from src.agents.player_template import PlayerTemplate


def extract_state(player, round_state):
    return (
        player.stack,
        round_state["pot"]["main"]["amount"],
        len(round_state["community_card"])
    )


class AdaptivePlayer(PlayerTemplate):

    def __init__(self, agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.episode = []

    def declare_action(self, valid_actions, hole_card, round_state):
        state = extract_state(self, round_state)

        action_id = self.agent.act(state, valid_actions)

        action_info = valid_actions[action_id]
        action = action_info['action']
        # action_id == 2 -> raise
        amount = action_info['amount']["min"] if action_id == 2 else action_info['amount']

        self.episode.append((state, action_id))
        return action, amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        final_stack = self.stack
        reward = final_stack - self.initial_stack

        for state, action in self.episode:
            self.agent.store((state, action, reward))

        self.agent.learn()
        self.episode = []
        print(f"AdaptivePlayer reward = {reward}")
