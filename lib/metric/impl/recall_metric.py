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

    def _calculate(self, pred_values, true_values, ctx):
        true_values = torch.round(true_values, decimals=self._rating_decimals)
        pred_values = torch.round(pred_values, decimals=self._rating_decimals)

        return recall_score(true_values, pred_values, average=self._average, zero_division=0)
2