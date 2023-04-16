from ..mean_user_at_k_metric import MeanUserAtkMetric
from metric.discretizer import identity
import torch
import math


def discount_cumulative_gain(ratings):
    return sum([r / math.log((i+1)+1) for i, r in enumerate(ratings.float())])


class MeanNdcgAtk(MeanUserAtkMetric):
    def __init__(self, user_index=0, k=10, decimals=4, rating_decimals=0):
        super().__init__('mNDCG' , user_index, k, decimals, identity(), rating_decimals)

    def _score(self, y_pred, y_true):
        y_true_ordered_index = torch.argsort(y_true, descending=True)
        y_pred_ordered_index = torch.argsort(y_pred, descending=True)

        y_true_ordered_by_pred = y_true[y_pred_ordered_index]
        y_true_ordered         = y_true[y_true_ordered_index]


        return discount_cumulative_gain(y_true_ordered_by_pred) / discount_cumulative_gain(y_true_ordered)
