import sys
sys.path.append('./lib')

import pytest
import model as ml
import data as dt
import torch
from utils import *


class TestKNNUserBasedPredictor:
    def test_when_request_3_neighbors_it_gets_3_nearest(self):
        # Prepare
        rm = dt.RatingsMatrix.from_tensor(torch.tensor([
            [2., 2., 4.], # a
            [4., 4., 3.], # b
            [1., 2., 2.], # c
            [3., 1., 1.]  # d
        ]))
        # a-a: 0
        # Nearest:
        #   a-c: 3
        #   a-b or a-d: 5

        predictor = ml.KNNUserBasedPredictor(
            rm, 
            distance=ml.CosineDistance(), 
            n_neighbors=2
        )
        user_idx = 0
        item_idx = 0

        # Perform
        predicted_rating = predictor.predict(user_idx, item_idx, debug=True)

        # Asserts
        assert 2.4837491512298584 == predicted_rating


