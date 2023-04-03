import torch
from torch.nn import Module, ReLU, Embedding, Linear, Dropout
from pytorch_common.modules import CommonMixin, PredictMixin
import logging
from ..mlp import MultiLayerPerceptron


class CollaborativeDenoisingEncoder(Module, CommonMixin, PredictMixin):
    def __init__(
        self,
        n_users,
        n_item_ratings,
        ratings_hidden_units = [],
        noise_dropout        = 0.2,
        mpl_activation       = ReLU(),
        mpl_batch_norm       = True,
        mpl_dropout          = 0.2,
        latent_space_dim     = 256
    ):
        super().__init__()
        self.type = 'CollaborativeDenoisingEncoder'

        # Encoder...
        self.users_embedding          = Embedding(n_users, latent_space_dim)
        self.noise_layer              =  Dropout(noise_dropout)

        self.items_ratings_mlp        = Linear(n_item_ratings, latent_space_dim, bias=True)

        units_per_layer = [n_item_ratings] + ratings_hidden_units + [latent_space_dim]
        self.mlp = MultiLayerPerceptron(
            units_per_layer = units_per_layer,
            activation      = [mpl_activation]  * (len(units_per_layer)-1),
            batch_norm      = [mpl_batch_norm]  * (len(units_per_layer)-1),
            dropout         = [mpl_dropout]     * (len(units_per_layer)-1)
        )

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

        # Reduce item_ratings dimensionality...
        item_ratings_embed = self.mlp(noisy_item_ratings)

        output = torch.add(users_embed, item_ratings_embed)

        if verbose:
            logging.info(f'{self.type} - Output: {output.shape}')
        return output