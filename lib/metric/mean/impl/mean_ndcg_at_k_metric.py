from ..mean_user_at_k_metric import MeanUserAtkMetric
from metric.discretizer import identity
import torch


class Metrics:
    @classmethod
    def idiscount_cumulative_gain(clazz, ratings, descendent=True):
        descendent_ratings = sorted(ratings, reverse=descendent)

        ratings_set = list(set(ratings))
        if len(ratings_set) == 1 and ratings_set[0] <= 2:
            return (8 - ratings_set[0])

        return clazz.discount_cumulative_gain(descendent_ratings)


    @staticmethod
    def discount_cumulative_gain(ratings):
        return sum([float(r) / math.log(i+2, 2) for i, r in enumerate(ratings)])

    @classmethod
    def normalized_discount_cumulative_gain(clazz, ratings, descendent=True):
        return clazz.discount_cumulative_gain(ratings) / clazz.idiscount_cumulative_gain(ratings, descendent)


class MeanNdcgAtk(MeanUserAtkMetric):
    def __init__(self, user_index=0, k=10, decimals=4, discretizer=identity(), rating_decimals=0):
        super().__init__('mNDCG' , user_index, k, decimals, discretizer, rating_decimals)

    def _score(self, y_pred, y_true):
        return normalized_discount_cumulative_gain(y_true[:self.k])
