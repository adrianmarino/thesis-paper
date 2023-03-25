import torch
from torch.nn import Module
from .encoder import Encoder
from .decoder import Decoder
from pytorch_common.callbacks import CallbackManager
from pytorch_common.callbacks import CallbackManager
from pytorch_common.modules import CommonMixin, PredictMixin, PersistentMixin
from torch.utils.data import DataLoader


class AutoEncoder(Module, CommonMixin, PredictMixin, PersistentMixin):
    def __init__(self, data_size, intermediate_size=1000, encoded_size=100, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(data_size, intermediate_size, encoded_size, dropout)
        self.decoder = Decoder(encoded_size, intermediate_size, data_size, dropout)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, inputs):
        encoded_representation = self.encoder(inputs)
        inputs_reconstruction = self.decoder(encoded_representation)
        return inputs_reconstruction

    def encode_from_data_loader(self, data_loader: DataLoader):
        self.eval()
        return torch.vstack([self.encoder(feat.to(self.device)) for (feat, _) in data_loader])

    def encode_from_batch(self, input_batch):
        self.eval()
        return torch.vstack([self.encoder(input_batch.to(self.device).float())])

