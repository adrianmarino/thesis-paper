import torch
from torch.nn import Module, Linear, ReLU
from pytorch_common.modules import CommonMixin
import logging
from ..mlp import MultiLayerPerceptron


class CollaborativeDecoder(Module, CommonMixin):
    def __init__(
        self,
        n_item_ratings,
        latent_space_dim = 256,
        activation       = ReLU(),
        dropout          = 0.2,
        batch_norm       = True,
        hidden_units     = []
    ):
        super().__init__()
        self.type = 'CollaborativeDecoder'

        units_per_layer = [latent_space_dim] + hidden_units + [n_item_ratings]
        self.mlp = MultiLayerPerceptron(
            units_per_layer = units_per_layer,
            activation      = [activation]  * (len(units_per_layer)-1),
            batch_norm      = [True]        * (len(units_per_layer)-1),
            dropout         = [dropout]     * (len(units_per_layer)-1)
        )


    def forward(self, input_data, verbose=False):
        """
        input  = [batch_size, [latent_space_dim]]
        output = [batch_size, [n_item_ratings]]
        """
        return self.mlp(input_data.to(self.device))