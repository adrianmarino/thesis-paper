import sys
sys.path.append('./lib')

import warnings
warnings.filterwarnings('ignore')

import pytest
import model as ml
import torch
from pytorch_common.util import tensor_eq
from torch import FloatTensor, Tensor


def tensor_round(tensor, n_digits=2):
    return torch.round(tensor * 10 ** n_digits) / (10 ** n_digits)


class TestEmbeddingFactorizationMachine:
    def test_forward_one_feature_vector(self):
        # Prepare
        embeddings = FloatTensor([
            [  # Feature vector
                [0.1, 0.2],  # Feature 1 embedding vector
                [0.2, 0.3]  # Feature 2 embedding vector
            ]
        ])
        layer = ml.EmbeddingFactorizationMachine()

        # Perform
        y = layer(embeddings)

        # Asserts
        assert y.shape == torch.Size((1, 1))
        assert tensor_eq(tensor_round(y), Tensor([[0.08]]))
