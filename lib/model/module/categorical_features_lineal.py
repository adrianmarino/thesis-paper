from torch import sum, zeros
from torch.nn import Parameter, Module
from torch.nn.init import normal_
from pytorch_common.modules import CommonMixin
from .multi_feature_embedding import MultiFeatureEmbedding


class CategoricalFeaturesLineal(Module, CommonMixin):
    def __init__(self, features_n_values: list[int], n_outputs: int=1, sparse: bool=False):
        super().__init__()
        self.embedding = MultiFeatureEmbedding(
            features_n_values = features_n_values,
            embedding_size    = n_outputs,
            sparse            = sparse
        )
        self.bias = Parameter(zeros((n_outputs,)))
        normal_(self.bias.data)

    def forward(self, x): return sum(self.embedding(x), dim=1) + self.bias
