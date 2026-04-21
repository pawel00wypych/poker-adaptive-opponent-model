import random

class SimpleAgent:

    def __init__(self):
        self.memory = []

    def act(self, state, valid_actions):
        return random.choice(range(len(valid_actions)))

    def store(self, transaction):
        self.memory.append(transaction)

    def learn(self):
        # TODO
        pass