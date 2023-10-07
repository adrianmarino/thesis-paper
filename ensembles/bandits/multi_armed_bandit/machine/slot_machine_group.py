import numpy as np


class SlotMachineGroup:
    def __init__(self):
        self.__machines = []
        self.name = 0

    def add(self, machine):
        self.name += 1
        machine.name = self.name
        self.__machines.append(machine)
        return self

    @property
    def gain_probs(self):
        return [b.gain_prob for b in self.__machines]

    @property
    def names(self):
        return [b.name for b in self.__machines]

    @property
    def attempts(self):
        return [b.attempts for b in self.__machines]

    @property
    def gain(self):
        return [b.gain for b in self.__machines]

    @property
    def loss(self):
        return [b.loss for b in self.__machines]

    def select_machine(self, strategy):
        return strategy.select(self)

    @property
    def machines(self):
        return self.__machines

    def __len__(self):
        return len(self.__machines)

    @property
    def max_gain_machine(self):
        return self.machines[np.argmax(self.gain_probs)]
