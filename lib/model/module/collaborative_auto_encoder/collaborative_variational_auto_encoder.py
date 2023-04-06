import torch
import logging

from torch.nn               import Module, Linear, ReLU
from pytorch_common.modules import CommonMixin, FitMixin, PersistentMixin

from .collaborative_variational_encoder import CollaborativeVariationalEncoder
from .collaborative_decoder             import CollaborativeDecoder
from .latent_space_encoder              import LatentSpaceEncoder


class CollaborativeVariationalAutoEncoder(Module, FitMixin, PersistentMixin):
    def __init__(
        self,
        n_item_ratings,
        encoder_dropout     = 0.2,
        encoder_activation  = ReLU(),
        mu_simgma_dim       : int = 512,
        latent_space_dim    : int  = 256
    ):
        super().__init__()
        self.type = 'CollaborativeVariationalAutoEncoder'

        self.encoder = CollaborativeVariationalEncoder(
            n_item_ratings,
            encoder_dropout,
            encoder_activation,
            mu_simgma_dim,
            latent_space_dim
        )
        self.decoder = CollaborativeDecoder(n_item_ratings, latent_space_dim)


    def forward(self, input_data, verbose=False):
        """
        input_data   = [batch_size, n_item_ratings]
        outputd_data = [batch_size, n_item_ratings]
        """
        latent_space = self.encoder(input_data, verbose)
        return self.decoder(latent_space, verbose)


    def as_encoder(self):
        return LatentSpaceEncoder(self)