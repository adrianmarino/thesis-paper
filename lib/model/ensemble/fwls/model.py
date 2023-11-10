import torch
from itertools import chain
from pytorch_common.modules import FitMixin, PredictMixin, PersistentMixin
import pytorch_common.util as pu
from .function import FeatureWeightFunction



class FeatureWeightLinearStacking(torch.nn.Module, FitMixin, PredictMixin, PersistentMixin):
    def __init__(self, predictors, *conditions, device=pu.get_device()):
        super().__init__()
        self.device_ = device
        self.predictors = predictors
        self.feature_weight_fns = [FeatureWeightFunction(*conditions).to(self.device) for _ in self.predictors]


    def forward(self, user_item_batch):
        terms = []

        for idx, predictor in enumerate(self.predictors):
            fw_fn = self.feature_weight_fns[idx]

            predictions = predictor(user_item_batch).to(self.device)

            fw_fn_value = fw_fn(user_item_batch)

            terms.append(fw_fn_value * predictions)

        return torch.sum(torch.stack(terms).T, dim=1)

    @property
    def device(self):
        return self.device_

    def parameters(self):
        return list(chain(*[fwfn.parameters() for fwfn in self.feature_weight_fns]))