# -----------------------------------------------------------------------
# Read: https://markelsanz14.medium.com/introducci%C3%B3n-al-aprendizaje-por-refuerzo-parte-1-el-problema-del-bandido-multibrazo-afe05c0c372e
# -----------------------------------------------------------------------
import multi_armed_bandit as mmb

#
#
#
#
# -----------------------------------------------------------------------
# Hyperparameters...
# -----------------------------------------------------------------------
params = mmb.ExperimentParams(
    success_probs=[0.1, 0.3, 0.005, 0.55, 0.4],
    epsilon=0.1,
    epsilon_delta=0.002,
    n_iterations=70
)
#
#
#
#
# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

group = mmb.SlotMachineGroup()

for p in params.success_probs:
    group.add(mmb.DefaultSlowMachine(p, reward=params.reward))

selection_strategy = mmb.EpsilonGreedySelectionStrategy(
    params.epsilon,
    params.warmup
)

result = mmb.experiment(group, selection_strategy, params)

result.show()
