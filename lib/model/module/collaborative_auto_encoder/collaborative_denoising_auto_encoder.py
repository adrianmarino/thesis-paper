import torch
import logging

from torch.nn               import Module, Linear, ReLU
from pytorch_common.modules import CommonMixin, FitMixin, PersistentMixin

from .collaborative_denoising_encoder   import CollaborativeDenoisingEncoder
from .collaborative_decoder             import CollaborativeDecoder
from .latent_space_encoder              import LatentSpaceEncoder


class CollaborativeDenoisingAutoEncoder(Module, FitMixin, PersistentMixin):
    def __init__(
        self,
        n_users,
        n_item_ratings,
        encoder_hidden_units = [],
        encoder_dropout              = 0.2,
        encoder_batch_norm           = True,
        encoder_activation           = ReLU(),
        decoder_hidden_units         = [],
        decoder_dropout              = 0.2,
        decoder_batch_norm           = True,
        decoder_activation           = ReLU(),
        latent_space_dim             = 256
    ):
        super().__init__()
        self.type = 'CollaborativeDenoisingAutoEncoder'

        self.encoder = CollaborativeDenoisingEncoder(
            n_users,
            n_item_ratings,
            encoder_hidden_units,
            encoder_activation,
            encoder_dropout,
            encoder_batch_norm,
            latent_space_dim
        )
        self.decoder = CollaborativeDecoder(
            n_item_ratings,
            latent_space_dim,
            decoder_activation,
            decoder_dropout,
            decoder_batch_norm,
            decoder_hidden_units
        )


    def forward(self, input_data, verbose=False):
        """
        input_data = (
            [batch_size, user_ids],
            [batch_size, [n_item_ratings]]
        )
        return [batch_size, [n_item_ratings]]
        """
        latent_space = self.encoder(input_data, verbose)
        return self.decoder(latent_space, verbose)


    def as_encoder(self):
        return LatentSpaceEncoder(self)