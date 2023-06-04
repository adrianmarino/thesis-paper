import sys
sys.path.append('./lib')

import warnings
warnings.filterwarnings('ignore')

import pytest
import model as ml
from torch import LongTensor, Size


class TestCategoricalFeaturesLineal:
    def test_create_lineal_for_two_categorical_features(self):
        # Prepare...
        cat_feature_1_possible_values = 2
        cat_feature_2_possible_values = 5
        output = 1

        # Perform...
        layer = ml.CategoricalFeaturesLineal(
            [cat_feature_1_possible_values, cat_feature_2_possible_values],
            output
        )

        # Asserts...
        weights_shape = layer.params['embedding.embedding.weight'].shape
        bias_shape = layer.params['bias'].shape

        assert weights_shape[0] == (cat_feature_1_possible_values + cat_feature_2_possible_values)
        assert weights_shape[1] == output
        assert bias_shape[0] == output

    def test_forward_two_feature_vectors(self):
        # Prepare...
        cat_feature_1_possible_values = 2
        cat_feature_2_possible_values = 5
        n_output = 1

        layer = ml.CategoricalFeaturesLineal(
            [cat_feature_1_possible_values, cat_feature_2_possible_values],
            n_output
        )
        X = LongTensor([
            [0, 0],  # Features vector 1
            [1, 2]  # Features vector 2
        ])

        # Perform...
        y = layer(X)

        # Asserts...
        assert y.shape == Size([y.shape[0], n_output])