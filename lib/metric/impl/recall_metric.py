from ..abstract_metric import AbstractMetric
from sklearn.metrics import recall_score
import torch


class RecallMetric(AbstractMetric):
    def __init__(self, decimals=4, average='macro', rating_decimals=0):
        name = f'Recall'
        if average != 'macro':
            name += f'({average})'
 
        super().__init__(name, decimals)
        self._average = average
        self._rating_decimals = rating_decimals

    def _calculate(self, y_pred, y_true, X):
        rounded_y_true = torch.round(y_true, decimals=self._rating_decimals)
        rounded_y_pred = torch.round(y_pred, decimals=self._rating_decimals)

        return recall_score(rounded_y_true, rounded_y_pred, average=self._average, zero_division=0)
2