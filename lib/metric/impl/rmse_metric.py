from ..abstract_metric import AbstractMetric
from sklearn.metrics import mean_squared_error
from math import sqrt


class RMSEMetric(AbstractMetric):
    def __init__(self, decimals=4):
        super().__init__('RMSE', decimals)

    def _calculate(self, y_pred, y_true, X):
        return sqrt(mean_squared_error(y_true, y_pred))
