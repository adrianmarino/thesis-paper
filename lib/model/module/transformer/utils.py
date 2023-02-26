import torch
from torch import Tensor


def generate_square_subsequent_mask(size: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

 