from ..mean_user_at_k_metric import MeanUserAtkMetric
from metric.discretizer import identity
import torch


def average_precision(y_pred, y_true):
    tps       = (y_pred == y_true).int()
    tps_count = torch.sum(tps)
    k         = y_true.shape[0]

    return sum([tp / (i+1) for i, tp in enumerate(tps)]) / min(k, tps_count) if tps_count > 0 else torch.tensor(0.0)


class MeanAveragePrecisionAtk(MeanUserAtkMetric):
    def __init__(self, user_index=0, k=10, decimals=4, discretizer=identity(), rating_decimals=0):
        super().__init__('mAP' , user_index, k, decimals, discretizer, rating_decimals)

    def _score(self, y_pred, y_true):
        return average_precision(y_pred, y_true)