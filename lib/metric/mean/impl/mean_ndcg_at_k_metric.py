from ..mean_user_at_k_metric import MeanUserAtkMetric
from metric.discretizer import identity
import torch
import math
from metric.common import dcg


class MeanNdcgAtk(MeanUserAtkMetric):
    def __init__(self, user_index=0, k=10, decimals=4, rating_decimals=0):
        super().__init__('mNDCG' , user_index, k, decimals, identity(), rating_decimals)

    def _score(self, y_pred, y_true):
        y_true_ordered_index = torch.argsort(y_true, descending=True)
        y_pred_ordered_index = torch.argsort(y_pred, descending=True)

        y_true_ordered_by_pred = y_true[y_pred_ordered_index]
        y_true_ordered         = y_true[y_true_ordered_index]


        return dcg(y_true_ordered_by_pred.float()) / dcg(y_true_ordered.float())
