import model as ml
import pytorch_common.util as pu
from torch import nn, Tensor
import math
from pytorch_common.modules import FitMixin
import logging
import torch


class TransformerClasifier(nn.Module, FitMixin):
    def __init__(
        self,
        embedding_weights    = None,
        vocab_size           = None,
        embedding_dim        = None,
        mask_position        : int   = 0,
        sequence_size        : int   = 10,
        nhead                : int   = 2,
        hidden_state_size    : int   = 200,
        n_transformer_layers : int   = 2,
        dropout              : float = 0.2,
        n_classes            : int   = 2
    ):
        """
        Create a Transformer Clasifier.

        Args:
            - embedding_weights: Feature embeddings.
            - nhead: Number of heads in nn.MultiheadAttention
            - hidden_state_size : Dimension of the feedforward network model in nn.TransformerEncoder.
            - n_transformer_layers: Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            - dropout: Dropout probability.
        """
        super().__init__()
        self.model_type = 'TransformerClasifier'
        self.mask_position = mask_position
        self.sequence_size = sequence_size

        if embedding_dim == None and vocab_size == None:
            self.embedding_dim      = embedding_weights.shape[1] 
            vocab_size              = embedding_weights.shape[0]
            self.embedding_encoder  = ml.EmbeddingLayerFactory.create_from_weights(embedding_weights)
        else:
            self.embedding_dim      = embedding_dim
            self.embedding_encoder  = ml.EmbeddingLayerFactory.create(vocab_size, embedding_dim)

        self.positional_encoder = ml.PositionalEncoding(
            self.embedding_dim, 
            vocab_size,
            dropout
        )

        self.transformer_encoder = ml.TransformerEncoderFactory.create(
            self.embedding_dim,
            nhead, 
            hidden_state_size,
            dropout,
            n_transformer_layers
        )
        
        self.mlp_decoder = ml.LinearUtils.init_weights(
            nn.Linear(self.embedding_dim, n_classes)
        )


    def forward(self, src: Tensor, verbose=False) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
            src_key_padding_mask: Tensor, shape [seq_len, batch_size] bool (True is paded token)
        Returns:
            output: Tensor, shape [batch_size, class_probabilities]
        """    
        seq_len    = src.shape[1]

        src = src.to(self.device)
        if verbose:
            logging.info(f'src: {src.shape}')

        try:
            embed = self.embedding_encoder(src) * math.sqrt(self.embedding_dim)
        except Exception as e:
            print(src)
            raise e

        if verbose:
            logging.info(f'embed: {embed.shape}')

        pos_embed = self.positional_encoder(embed)
        if verbose:
            logging.info(f'pos_embed: {pos_embed.shape}')


        src_mask = ml.generate_square_subsequent_mask(seq_len).to(self.device)
        if verbose: 
            logging.info(f'src_mask: {src_mask.shape}')


        src_key_padding_mask = (src == self.mask_position)
        if verbose:
            logging.info(f'src_key_padding_mask: {src_key_padding_mask.shape}')

        trans_input = pos_embed.permute(1, 0, 2)
        if verbose:
            logging.info(f'trans_input: {trans_input.shape}')

        trans_output = self.transformer_encoder(
            trans_input,
            src_key_padding_mask = src_key_padding_mask.to(pu.get_device())
        )

        trans_output = trans_output.permute(1, 0, 2)
        if verbose:
            logging.info(f'trans_output: {trans_output.shape}')

        mlp_input = trans_output[:, -1]
        if verbose:
            logging.info(f'mlp_input: {mlp_input.shape}')


        mlp_output = self.mlp_decoder(mlp_input)
        if verbose:
            logging.info(f'mlp_output: {mlp_output.shape}')

        return mlp_output 