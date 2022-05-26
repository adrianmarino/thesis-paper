import torch
from pytorch_common.modules import FitMixin
import model as ml


class GMF(torch.nn.Module, FitMixin):
    def __init__(self,
            n_users,
            n_items,
            embedding_size,
            sparse=False
    ):
        super().__init__()
        self.embedding = ml.MultiFeatureEmbedding(
            features_n_values = [n_users, n_items], 
            embedding_size    = embedding_size,
            sparse            = sparse
        )
        self.dot = ml.BatchDot()

    def forward(self, x_batch):
        # Lookup embedding vectors by users and items index...
        x_emb_batch = self.embedding(x_batch)

        # Get users and items embedding vectors... 
        users = x_emb_batch[:, 0].unsqueeze(1)
        items = x_emb_batch[:, 1].unsqueeze(1)

        # Batch dot product...
        return self.dot(users, items)