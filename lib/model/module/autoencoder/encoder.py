import torch
from .layer_group import LayerGroup


class Encoder(torch.nn.Module):
    def __init__(self, input_size, intermediate_size, encoding_size, dropout=0.2):
        super(Encoder, self).__init__()
        layers = LayerGroup.linearBatchNormReluDropout(input_size, intermediate_size, dropout) + \
                 LayerGroup.linearBatchNormReluDropout(intermediate_size, encoding_size, dropout)
        self.mlp = Sequential(*layers)

    def forward(self, x): return self.mlp(x)
