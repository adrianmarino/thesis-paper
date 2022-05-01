from abc import ABC
import torch


class AbstractMetric(ABC):
    def __init__(self, name, decimals=4):
        self.name = name
        self._decimals = decimals

    def perform(self, pred_values, true_values, opts={}):
        metric = self._calculate(pred_values, true_values, opts)
        return torch.round(torch.tensor(metric), decimals=self._decimals)

    def _calculate(self, pred_values, true_values, opts):
        pass
