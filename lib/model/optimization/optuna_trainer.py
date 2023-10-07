import optuna


class HyperParamsSampler:
    def sample(self, trial):
        pass

class OptunaTrainer:
    def __init__(
        self, 
        params_sampler,
        train_fn,
        train_dl,
        eval_dl,
        eval_metric
):
        self.__params_sampler = params_sampler
        self.__train_fn       = train_fn
        self.__train_dl       = train_dl
        self.__eval_dl        = eval_dl
        self.__eval_metric    = eval_metric

    def __call__(self, trial):
        hyper_params = self.__params_sampler.sample(trial)

        model = self.__train_fn(self.__train_dl, hyper_params)

        metric_value = self.__eval_metric(*model.evaluate(self.__eval_dl))

        trial.report(metric_value, hyper_params.epochs)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return metric_value