import torch
import pytorch_common.util as pu


class CosineDistance(torch.nn.Module):
    def __init__(self, dim=1, device=pu.get_device()):
        super().__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=dim).to(device)
        self.device = device

    def forward(self, a, b):
        return 1 - self.similarity(a.to(self.device), b.to(self.device))


