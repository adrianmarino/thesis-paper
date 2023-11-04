from multi_armed_bandit.machine.slot_machine import SlowMachine


class DefaultSlowMachine(SlowMachine):

    def __init__(self, success_prob, reward=5, loss_points=1):
        super().__init__(success_prob, reward, loss_points)

    def compute_success_probability(self, is_success):
        return self.gain / float(self.attempts)
