import torch
from torch.nn import Module, Sequential
from .layer_group import LayerGroup


class Decoder(Module):
    def __init__(self, encoding_size, intermediate_size, output_size, dropout=0.2):
        super().__init__()
        layers = LayerGroup.linearBatchNormReluDropout(encoding_size, intermediate_size, dropout) + \
                 LayerGroup.linearBatchNormSigmoid(intermediate_size, output_size)
        self.mlp = Sequential(*layers)

    def forward(self, x): return self.mlp(x)

