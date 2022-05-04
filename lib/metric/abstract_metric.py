from abc import ABC
import torch


class AbstractMetric(ABC):
    def __init__(self, name, decimals=4):
        self.name = name
        self._decimals = decimals

    def perform(self, y_pred, y_true, X):
        metric = self._calculate(y_pred, y_true, X)
        return torch.round(metric, decimals=self._decimals)

    def _calculate(self, y_pred, y_true, X):
        pass
