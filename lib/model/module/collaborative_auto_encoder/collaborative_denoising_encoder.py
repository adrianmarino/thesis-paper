import torch
from torch.nn import Module, ReLU, Embedding, Linear, Dropout
from pytorch_common.modules import CommonMixin, PredictMixin
import logging


class CollaborativeDenoisingEncoder(Module, CommonMixin, PredictMixin):
    def __init__(
        self,
        n_users,
        n_item_ratings,
        activation        = None,
        dropout           = 0.2,
        latent_space_dim  : int  = 256
    ):
        super(CollaborativeDenoisingEncoder, self).__init__()
        self.type = 'CollaborativeDenoisingEncoder'

        # Encoder...
        self.users_embedding          = Embedding(n_users, latent_space_dim)
        self.items_ratings_linear     = Linear(n_item_ratings, latent_space_dim, bias=True)
        self.noise_layer              =  Dropout(dropout)
        self.items_ratings_activation = activation


    def forward(self, input_data, verbose=False):
        """
        input  = (
            [batch_size, user_ids],
            [batch_size, [n_item_ratings]]
        )
        output = [batch_size, [latent_space_dim]]
        """

        # Get inputs...
        user_ids     = input_data[:, :1].squeeze(1).int().to(self.device)
        item_ratings = input_data[:, 1:].float().to(self.device)
        if verbose:
            logging.info(f'{self.type} - Input - Users: {user_ids.shape} - Ratings: {item_ratings.shape}')

        # Resolver users embeddings...
        users_embed = self.users_embedding(user_ids)

        # Apply noise to item_ratings...
        noisy_item_ratings = self.noise_layer(item_ratings)

        # Reduce item_ratins dimensionality...
        item_ratings_embed = self.items_ratings_linear(noisy_item_ratings)

        # Aplli
        if self.items_ratings_activation:
            item_ratings_embed = self.items_ratings_activation(item_ratings_embed)

        output = users_embed + item_ratings_embed
        if verbose:
            logging.info(f'{self.type} - Output: {output.shape}')
        return output