import sys
sys.path.append('./lib')

import pytest
import model as ml
import data as dt
import torch
from utils import *


class TestKNNUserBasedPredictor:
    def test_when_predict_rating_made_buy_a_user_for_a_given_item_it_returns_a_valid_value(self):
        # Prepare
        rm = dt.RatingsMatrix.from_tensor(torch.tensor([
            [2., 2., 4.], # a
            [4., 4., 3.], # b
            [1., 2., 2.], # c
            [3., 1., 1.]  # dQ
        ]))
        # a-a: 0
        # Nearest:
        #   a-c: 3
        #   a-b or a-d: 5

        predictor = ml.KNNUserBasedPredictor(
            rm, 
            distance=ml.CosineDistance()
        )
        user_idx = 0
        item_idx = 0

        # Perform
        predicted_rating = predictor.predict(
            user_idx, 
            item_idx, 
            n_neighbors=2,
            debug=True
        )

        # Asserts
        assert 2.4837491512298584 == predicted_rating


