from ..mean_user_at_k_metric import MeanUserAtkMetric
from metric.discretizer import identity
import torchmetrics


class MeanUserFBetaScoreAtk(MeanUserAtkMetric):
    def __init__(self, user_index=0, k=10, decimals=4, discretizer=identity(), rating_decimals=0, n_classes=None, average='micro', beta=1.0):
        name = f'MeanUserF{beta}Score'
        if average != 'micro':
            name += f'({average})'

        super().__init__(name, user_index, k, decimals, discretizer, rating_decimals)

        self.__metric = torchmetrics.FBetaScore(
            task        = 'multiclass',
            num_classes = n_classes,
            average     = average,
            beta        = beta
        )


    def _score(self, y_pred, y_true):
        return self.__metric(y_pred, y_true)
