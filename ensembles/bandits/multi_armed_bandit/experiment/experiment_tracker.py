import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class ExperimentTracker:
    def __init__(self, slot_machine_group, num_iterations):
        self.slot_machine_group = slot_machine_group
        self.num_iterations = num_iterations
        self.logs = []

    def save_logs(self, iteration):
        self.logs.extend([{
            'iteration': iteration,
            'machine': machine.name,
            'rewards': machine.gain,
            'loss': machine.loss
        } for machine in self.slot_machine_group.machines])

    def show(self):
        [print(b) for b in self.slot_machine_group.machines]

        best_machine = self.slot_machine_group.max_gain_machine
        print(f"""
    - Investment: {self.num_iterations} points.
    - Final Points: {sum(self.slot_machine_group.gain) - sum(self.slot_machine_group.loss)} points.
    - Max Gain Machine: {best_machine.name}
        """)

    def plot(self):
        sns.lineplot(data=pd.DataFrame(self.logs), x='iteration', y='rewards', hue='machine') \
            .set_title('Money Gain by Machine')
        plt.show()

        sns.lineplot(data=pd.DataFrame(self.logs), x='iteration', y='loss', hue='machine') \
            .set_title('Money Lost by Machine')
        plt.show()
