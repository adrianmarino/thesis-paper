# -----------------------------------------------------------------------
# Read: https://visualstudiomagazine.com/articles/2019/06/01/thompson-sampling.aspx
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
    epsilon=0.11,
    epsilon_delta=0.001,
    n_iterations=45,
    seed=42,
    warmup=10
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
    group.add(mmb.ThomsonSamplingSlowMachine(
        p,
        seed=params.seed,
        reward=params.reward
    ))

selection_strategy = mmb.EpsilonGreedySelectionStrategy(
    params.epsilon,
    params.warmup
)

result = mmb.experiment(group, selection_strategy, params)

result.show()
