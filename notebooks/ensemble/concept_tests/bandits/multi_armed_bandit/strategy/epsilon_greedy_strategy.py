import numpy as np


class EpsilonGreedySelectionStrategy:
    def __init__(self, exploration_prob, warmup=0):
        self.exploration_prob = exploration_prob
        self.warmup = warmup

    def select(self, group):
        if self.warmup > 0 or np.random.normal() < self.exploration_prob:
            self.warmup -= 1
            return group.machines[np.random.randint(0, len(group.machines))]  # Random action
        else:
            return group.max_gain_machine  # Greedy action

    def decrease_exploration(self, delta):
        self.exploration_prob -= delta
