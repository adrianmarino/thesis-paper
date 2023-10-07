from abc import ABC, abstractmethod

import numpy as np


class SlowMachine(ABC):
    def __init__(self, success_prob, reward=5, loss_points=1):
        self.name = None
        self.__success_prob = success_prob
        self.__reward = reward
        self.__loss_points = loss_points
        self.gain = 0
        self.attempts = 0
        self.gain_prob = 0.0
        self.loss = 0

    def pull_arm(self):
        # random success
        #  - True: 1 reward or 1 success.
        #  - False: Zero rewards or no success.
        is_success = np.random.uniform() <= self.__success_prob
        self.attempts += 1

        if is_success:
            self.gain += self.__reward
        else:
            self.loss += self.__loss_points

        self.gain_prob = self.compute_success_probability(is_success)

    @abstractmethod
    def compute_success_probability(self, is_success):
        pass

    def __str__(self):
        real_gain = self.gain - self.loss
        if real_gain < 0:
            real_gain = 0

        real_loss = self.loss - self.gain
        if real_loss < 0:
            real_loss = 0

        real_gain_str = f"gain {real_gain} points" if real_gain > 0 else ""
        real_loss_str = f"loss {real_loss} points" if real_loss > 0 else ""

        reals = [s for s in [real_gain_str, real_loss_str] if len(s) > 0]

        real_str = ','.join(reals) if reals else 'gain == loss'

        return f"""
        Machine: {self.name}
        - Prob: {self.gain_prob:.4}
        - Attempts: {self.attempts}
        - {real_str.capitalize()}
        """
