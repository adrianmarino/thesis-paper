import torch
import models as ml


def vector_cos_dist(a, b): return ml.CosineDistance(dim=0)(a, b)

def equals(a: torch.Tensor, b: torch.Tensor): return torch.all(a == b)

def assert_equal_vector(a: torch.Tensor, b: torch.Tensor):  assert equals(a, b)
