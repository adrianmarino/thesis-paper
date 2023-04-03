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
        encoder_noise_dropout        = 0.2,
        encoder_mpl_dropout          = 0.2,
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
            n_users = n_users,
            n_item_ratings = n_item_ratings,
            ratings_hidden_units = encoder_hidden_units,
            noise_dropout        = encoder_noise_dropout,
            mpl_activation       = encoder_activation,
            mpl_batch_norm       = encoder_batch_norm,
            mpl_dropout          = encoder_mpl_dropout,
            latent_space_dim     = latent_space_dim
        )
        self.decoder = CollaborativeDecoder(
            n_item_ratings = n_item_ratings,
            latent_space_dim = latent_space_dim,
            activation       = decoder_activation,
            dropout          = decoder_dropout,
            batch_norm       = decoder_batch_norm,
            hidden_units     = decoder_hidden_units
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