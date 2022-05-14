import sys
sys.path.append('./lib')

import pytest
import model as ml
from utils import *
from torch import LongTensor, Size
from pytorch_common.util import tensor_eq


class TestMultiFeatureEmbedding:
    def test_create_embedding_for_two_categorical_features(self):
        # Prepare...
        cat_feature_1_possible_values = 2
        cat_feature_2_possible_values = 5
        embedding_size = 10

        # Perform...
        layer = ml.MultiFeatureEmbedding(
            [cat_feature_1_possible_values, cat_feature_2_possible_values],
            embedding_size
        )

        # Asserts...
        weights_shape = layer.params['embedding.weight'].shape
        assert weights_shape[0], cat_feature_1_possible_values + cat_feature_2_possible_values
        assert weights_shape[1], embedding_size

    def test_forward_two_feature_vectors(self):
        # Prepare...
        cat_feature_1_possible_values = 2
        cat_feature_2_possible_values = 5
        embedding_size = 10

        layer = ml.MultiFeatureEmbedding(
            [cat_feature_1_possible_values, cat_feature_2_possible_values],
            embedding_size
        )
        X = LongTensor([
            [0, 0],  # Features vector 1
            [1, 2]   # Features vector 2
        ])

        # Perform...
        y = layer(X)

        # Asserts...
        assert y.shape == Size([y.shape[0], y.shape[1], embedding_size])

        # Features vector 1
        assert tensor_eq(y[0][0], layer.vectors[0])
        assert tensor_eq(y[0][1], layer.vectors[2])
        # Features vector 2
        assert tensor_eq(y[1][0], layer.vectors[1])
        assert tensor_eq(y[1][1], layer.vectors[4])
