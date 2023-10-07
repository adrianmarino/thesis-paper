import numpy as np

from multi_armed_bandit.machine.slot_machine import SlowMachine


class ThomsonSamplingSlowMachine(SlowMachine):

    def __init__(self, success_prob, seed=None, reward=5, loss_points=1):
        super().__init__(success_prob, reward, loss_points)
        self.n_success = 10
        self.n_fails = 0
        self.__rnd = np.random.RandomState(seed)

    def compute_success_probability(self, is_success):
        if is_success:
            self.n_success += 1
        else:
            self.n_fails += 1

        return self.__rnd.beta(self.n_success + 1, self.n_fails + 1, size=3)
