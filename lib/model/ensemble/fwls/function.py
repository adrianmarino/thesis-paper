import torch
from pytorch_common.modules import FitMixin, PredictMixin, PersistentMixin
from .conditions import ConditionContextFactory
import pytorch_common.util as pu


class FeatureWeightFunction(torch.nn.Module, FitMixin, PredictMixin, PersistentMixin):
    def __init__(self, *conditions):
        super(FeatureWeightFunction, self).__init__()
        self.conditions = list(conditions)
        self.weights = torch.nn.Parameter(torch.rand(len(self.conditions)))

    def __call__(self, user_item_batch, user_col=0):
        user_ids  = user_item_batch[:, user_col]

        ctx_factory = ConditionContextFactory(user_ids)

        values = []
        for user_id in user_ids:
            ctx = ctx_factory.create(user_id)
            values.append(torch.sum(torch.stack([weight * self.conditions[idx](ctx) for idx, weight in enumerate(self.weights)])))

        return torch.stack(values).to(self.device)