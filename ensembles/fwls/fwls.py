from abc import ABC

import torch


class FeatureWightCondition(ABC):
    def __call__(self, ctx):
        pass


class InteractionsRange(FeatureWightCondition):
    def __init__(self, begin=None, end=None):
        self.__begin = begin
        self.__end = end

    def __call__(self, ctx):
        if self.__begin and self.__end:
            return self.__begin <= ctx.n_user_interactions <= self.__end
        elif self.__begin and self.__end is None:
            return self.__begin <= ctx.n_user_interactions
        elif self.__begin is None and self.__end:
            return ctx.n_user_interactions <= self.__end
        else:
            raise Exception("Invalid range")


conditions = [InteractionsRange(end=3), InteractionsRange(begin=4)]


class FeatureWeightFunction(torch.nn.Module):
    def __init__(self, conditions):
        self.conditions = conditions
        self.weights = torch.rand(len(conditions))

    def forward(self, ratings):
        return sum([w * self.conditions[idx] for idx, w in enumerate(self.weights)]) * ratings


feature_weight_functions = [FeatureWeightFunction(conditions), FeatureWeightFunction(conditions)]


class FeatureWeightLinearStacking(torch.nn.Module):
    def __init__(self, predictors):
        super().__init__()
        self.feature_weight_fns = [FeatureWeightFunction(conditions) for _ in predictors]
        self.predictors = predictors

    def forward(self, user_item):
        return sum([self.feature_weight_fns[idx](model(user_item)) for idx, model in enumerate(self.models)])

class Predictor(ABC):
    def predict(self, x):
        pass


class StaticPredictor(Predictor):
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns
    def predict(self, x):





