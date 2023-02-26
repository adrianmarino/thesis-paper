import torch
from torch import nn
import numpy as np


class EmbeddingLayerFactory:
    @staticmethod
    def create_from_weights(weights):
        if type(weights) == np.ndarray:
            weights = torch \
                .from_numpy(weights) \
                .to(dtype=torch.float32)      

        layer = nn.Embedding(
            num_embeddings = weights.size(0),
            embedding_dim  = weights.size(1)
        )
        layer.weight = nn.Parameter(weights)
        return layer


    @staticmethod
    def create(num_embeddings, embedding_dim, initrange=0.1):
        layer = nn.Embedding(num_embeddings, embedding_dim)
        layer.weight.data.uniform_(-initrange, initrange)
        return layer