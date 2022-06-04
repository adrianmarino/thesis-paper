import torch
from .encoder import Encoder
from .decoder import Decoder
from pytorch_common.callbacks import CallbackManager
from pytorch_common.callbacks import CallbackManager
from pytorch_common.modules.common_mixin import CommonMixin


class AutoEncoder(torch.nn.Module, CommonMixin):
    def __init__(self, data_size, intermediate_size=1000, encoded_size=100, dropout=0.2):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(data_size, intermediate_size, encoded_size, dropout)
        self.decoder = Decoder(encoded_size, intermediate_size, data_size, dropout)

    def train(self):
        super.train()
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        super.eval()
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, inputs):
        encoded_representation = self.encoder(inputs)
        inputs_reconstruction = self.decoder(encoded_representation)
        return inputs_reconstruction

    def encoded_representation(self, inputs):
        self.eval()
        return self.encoder(inputs.to(self.device))
