from ..abstract_metric import AbstractMetric
from sklearn.metrics import precision_score
import torch


class PrecisionMetric(AbstractMetric):
    def __init__(self, decimals=4, average='macro', rating_decimals=0):
        name = f'Precision'
        if average != 'macro':
            name += f'({average})'

        super().__init__(name, decimals)
        self._average = average
        self._rating_decimals = rating_decimals

    def _calculate(self, pred_values, true_values, opts):
        true_values = torch.round(true_values, decimals=self._rating_decimals)
        pred_values = torch.round(pred_values, decimals=self._rating_decimals)

        return precision_score(true_values, pred_values, average=self._average, zero_division=0)
