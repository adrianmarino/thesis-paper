import sys
sys.path.append('./lib')

import warnings
warnings.filterwarnings('ignore')

import pytest
import model as ml
import torch


class TestDeepFM:
    def test_forward_one_batch_of_size_two(self):
        # Prepare...
        model = ml.DeepFM(
            features_n_values=[2, 3],
            embedding_size=4,
            units_per_layer=[20, 20, 20],
            dropout=0.2
        )
        X = torch.LongTensor([
            [0, 0],
            [1, 2]
        ])

        # Perform...
        y = model(X)

        # Asserts...
        assert y.shape[0] == 2
        for o in y:
            assert 0 >= 0