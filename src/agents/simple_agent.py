import random
import numpy as np

class SimpleAgent:

    def __init__(self):
        self.memory = []
        self.alpha = 0.1 # learning rate (how much does agent value the future)
        self.gamma = 0.9 # discount factor (gives more weight to immediate
        # reward
        self.epsilon = 0.5 # probability of exploration
        self.Q = {} # stack, pot, cards + 3 valid actions
        self.NUM_ACTIONS = 3  # fold, call, raise

    def act(self, state, valid_actions):
        # lazy state initialization
        if not state in self.Q:
            self.Q[state] = np.zeros(self.NUM_ACTIONS)

        # possible actions: 0 - fold, 1 - call, 2 - raise
        if np.random.random() < self.epsilon:
            return random.choice(range(len(valid_actions)))
        else:
            return int(np.argmax(self.Q[state]))

    def store(self, transaction):
        self.memory.append(transaction)

    def learn(self):
        # Episodic Monte Carlo control with ε-greedy policy (simplified)
        for state, action, reward in self.memory:
            if state not in self.Q:
                self.Q[state] = np.zeros(self.NUM_ACTIONS)

            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])

        self.memory = []
