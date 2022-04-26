import torch


class CosineDistance: 
    def __init__(self, dim=1): self.similarity = torch.nn.CosineSimilarity(dim=dim)    
    def __call__(self, a, b): return 1 - self.similarity(a, b)


