class ExperimentParams:
    def __init__(
            self,
            reward=1,
            success_probs=[0.1, 0.2],
            epsilon=0.13,
            epsilon_delta=0.0,
            n_iterations=50,
            seed=None,
            warmup=0
    ):
        self.reward = reward
        self.success_probs = success_probs
        self.epsilon = epsilon
        self.epsilon_delta = epsilon_delta
        self.n_iterations = n_iterations
        self.seed = seed
        self.warmup = warmup
