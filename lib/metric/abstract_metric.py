from abc import ABC
import torch


class AbstractMetric(ABC):
    def __init__(self, name, decimals=4):
        self.name = name
        self._decimals = decimals

    def perform(self, y_pred, y_true, X):
        metrics = self._calculate(y_pred, y_true, X)

        value = torch.round(metrics[0], decimals=self._decimals).item()
        metrics = (value,) + metrics[1:]

        return self._build_result(metrics)

    def _calculate(self, y_pred, y_true, X):
        pass

    def _build_result(self,  metrics):
        return {self.name: metrics[0]}