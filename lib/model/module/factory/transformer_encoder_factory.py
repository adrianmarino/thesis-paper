from torch import nn


class TransformerEncoderFactory:
    @staticmethod
    def create(
        embedding_dim, 
        nhead, 
        hidden_state_size, 
        dropout, 
        n_layers,
        activation = 'gelu'
    ):
        encoder_layers = nn.TransformerEncoderLayer(
            d_model         = embedding_dim,
            nhead           = nhead,
            dim_feedforward = hidden_state_size,
            dropout         = dropout,
            activation      = activation
        )
        return nn.TransformerEncoder(
            encoder_layers, 
            n_layers
        )
