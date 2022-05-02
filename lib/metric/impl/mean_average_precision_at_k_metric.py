from ..abstract_metric import AbstractMetric
from ..user_pred_true_sampler import UserYPredYTrueSampler
from ..binarizer import identity
import statistics
import torch
import logging


class MeanAveragePrecisionAtkMetric(AbstractMetric):
    def __init__(self, user_index, k=10, decimals=4, binarizer=identity(), rating_decimals=0):
        super().__init__(f'mAP@{k}{binarizer.desc}', decimals)
        self._sampler = UserYPredYTrueSampler(user_index, k, rating_decimals)
        self._binarize_fn = binarizer.closure()
        self._k = k

    def _calculate(self, y_pred, y_true, X):
        scores = [APK(y_true, self._k, self._binarize_fn).item() for _, y_true in self._sampler.sample(y_pred, y_true, X)]

        logging.debug(f'Users found: {len(scores)}')
        return statistics.mean(scores)


# AP@K
def APK(y_true, k, binarize_fn):
    n_true_pos = y_true.apply_(binarize_fn)
    tp_total   = torch.sum(n_true_pos)

    if tp_total == 0: return torch.tensor(0.0)

    return sum([v / (i+1) for i, v in enumerate(n_true_pos)]) / min(k, tp_total)
