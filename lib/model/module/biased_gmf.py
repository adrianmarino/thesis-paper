import torch
from pytorch_common.modules import FitMixin
import model as ml
import logging


class BiasedGMF(torch.nn.Module, FitMixin):
    def __init__(self,
            n_users: int,
            n_items: int,
            embedding_size: int,
            sparse:bool=False
    ):
        super().__init__()
        self.embedding = ml.MultiFeatureEmbedding(
            features_n_values = [n_users, n_items], 
            embedding_size    = embedding_size,
            sparse            = sparse
        )
        self.embedding_bias = ml.MultiFeatureEmbedding(
            features_n_values = [n_users, n_items], 
            embedding_size    = 1,
            sparse            = sparse
        )
        self.dot = ml.BatchDot()

    def forward(self, x_batch):
        # Lookup embedding vectors by users and items index...
        x_emb_batch = self.embedding(x_batch)
        
        x_emb_bias_batch = self.embedding_bias(x_batch)

        # Get users and items embedding vectors... 
        users = x_emb_batch[:, 0].unsqueeze(1)
        items = x_emb_batch[:, 1].unsqueeze(1)
        
        users_bias = x_emb_bias_batch[:, 0].squeeze(1)
        items_bias = x_emb_bias_batch[:, 1].squeeze(1)

        # Batch dot product...
        dot_product = self.dot(users, items)

        return dot_product + users_bias + items_bias