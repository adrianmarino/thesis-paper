import torch
import logging

from torch.nn               import Module, Linear, ReLU
from pytorch_common.modules import CommonMixin, FitMixin, PersistentMixin

from .collaborative_encoder             import CollaborativeEncoder
from .collaborative_decoder             import CollaborativeDecoder
from .latent_space_encoder              import LatentSpaceEncoder


class DeepAutoEncoder(Module, FitMixin, PersistentMixin):
    def __init__(
        self,
        n_item_ratings,
        latent_space_dim    : int  = 256,

        encoder_hidden_units         = [],
        encoder_dropout              = 0.2,
        encoder_batch_norm           = True,
        encoder_activation           = None,

        decoder_hidden_units         = [],
        decoder_dropout              = 0.2,
        decoder_batch_norm           = True,
        decoder_activation           = None,
    ):
        super().__init__()
        self.type = 'DeepAutoEncoder'
        self.encoder = CollaborativeEncoder(
            n_item_ratings   = n_item_ratings,
            latent_space_dim = latent_space_dim,
            activation       = decoder_activation,
            dropout          = decoder_dropout,
            batch_norm       = decoder_batch_norm,
            hidden_units     = decoder_hidden_units
        )
        self.decoder = CollaborativeDecoder(
            n_item_ratings   = n_item_ratings,
            latent_space_dim = latent_space_dim,
            activation       = decoder_activation,
            dropout          = decoder_dropout,
            batch_norm       = decoder_batch_norm,
            hidden_units     = decoder_hidden_units
        )


    def forward(self, input_data, verbose=False):
        """
        input_data   = [batch_size, n_item_ratings]
        outputd_data = [batch_size, n_item_ratings]
        """
        latent_space = self.encoder(input_data, verbose)
        return self.decoder(latent_space, verbose)


    def as_encoder(self):
        return LatentSpaceEncoder(self)