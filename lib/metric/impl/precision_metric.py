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

    def _calculate(self, y_pred, y_true, X):
        rounded_y_true = torch.round(y_true, decimals=self._rating_decimals)
        rounded_y_pred = torch.round(y_pred, decimals=self._rating_decimals)

        return precision_score(rounded_y_true, rounded_y_pred, average=self._average, zero_division=0)
