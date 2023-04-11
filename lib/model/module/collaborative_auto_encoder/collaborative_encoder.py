import torch
from torch.nn import Module, Linear, ReLU
from pytorch_common.modules import CommonMixin, PredictMixin
import logging
from ..mlp import MultiLayerPerceptron


class CollaborativeEncoder(Module, CommonMixin, PredictMixin):
    def __init__(
        self,
        n_item_ratings,
        latent_space_dim = 256,
        activation       = None,
        dropout          = 0.2,
        batch_norm       = True,
        hidden_units     = []
    ):
        super().__init__()
        self.type = 'CollaborativeEncoder'

        units_per_layer = [n_item_ratings] + hidden_units + [latent_space_dim]

        self.mlp = MultiLayerPerceptron(
            units_per_layer = units_per_layer,
            activation      = ([activation] if activation else [])  * (len(units_per_layer)-1),
            batch_norm      = ([batch_norm] if batch_norm else [])  * (len(units_per_layer)-1),
            dropout         = ([batch_norm] if batch_norm else [])  * (len(units_per_layer)-1)
        )


    def forward(self, input_data, verbose=False):
        """
        input  = [batch_size, [latent_space_dim]]
        output = [batch_size, [n_item_ratings]]
        """
        return self.mlp(input_data.to(self.device))