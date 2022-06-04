import torch
from .layer_group import LayerGroup


class Decoder(torch.nn.Module):
    def __init__(self, encoding_size, intermediate_size, output_size, dropout=0.2):
        super(Decoder, self).__init__()
        layers = LayerGroup.linearBatchNormReluDropout(encoding_size, intermediate_size, dropout) + \
                 LayerGroup.linearBatchNormSigmoid(intermediate_size, output_size)
        self.mlp = Sequential(*layers)

    def forward(self, x): return self.mlp(x)

