from ..abstract_metric import AbstractMetric
from sklearn.metrics import fbeta_score
import torch


class FBetaMetric(AbstractMetric):
    def __init__(self, beta=1, decimals=4, average='macro', rating_decimals=0):
        name = f'F{beta}Score'
        if average != 'macro':
            name += f'({average})'

        super().__init__(name, decimals)

        self._beta = beta
        self._average = average
        self._rating_decimals = rating_decimals

    def _calculate(self, pred_values, true_values, ctx):
        true_values = torch.round(true_values, decimals=self._rating_decimals)
        pred_values = torch.round(pred_values, decimals=self._rating_decimals)

        return fbeta_score(true_values, pred_values, beta=self._beta, average=self._average)
