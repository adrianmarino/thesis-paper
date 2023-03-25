import torch
from torch.nn import Module, Linear
from pytorch_common.modules import CommonMixin
import logging


class CollaborativeDecoder(Module, CommonMixin):
    def __init__(
        self,
        n_item_ratings,
        activation        = None,
        latent_space_dim  : int  = 256
    ):
        super(CollaborativeDecoder, self).__init__()
        self.type = 'CollaborativeDecoder'

        # Encoder...
        self.linear = Linear(latent_space_dim, n_item_ratings)
        self.activation = activation


    def forward(self, input_data, verbose=False):
        """
        input  = [batch_size, [latent_space_dim]]
        output = [batch_size, [n_item_ratings]]
        """

        if verbose:
            logging.info(f'{self.type} - Input: {input_data.shape}')

        item_ratings = self.linear(input_data.to(self.device))

        if self.activation:
            item_ratings = self.activation(item_ratings)

        if verbose:
            logging.info(f'{self.type} - Output: {item_ratings.shape}')

        return item_ratings