import sys
sys.path.append('./lib')

import pytest
import model as ml
import torch
from utils import *


class TestNearestNeighbors:
    def test_when_request_3_neighbors_it_gets_3_nearest(self):
        # Prepare
        matrix = torch.tensor([
            [2., 2., 4.], # a
            [4., 4., 3.], # b
            [1., 2., 2.], # c
            [3., 1., 1.]  # d
        ])
        # a-a: 0
        # Nearest:
        #   a-c: 3
        #   a-b or a-d: 5

        target  = ml.NearestNeighbors(matrix, ml.CosineDistance())
        row_idx = 0
        k       = 2

        # Perform
        result: NearestNeighborsResult = target.neighbors(row_idx, k)

        # Asserts
        assert len(result.rows) == len(result.distances) == len(result.indexes) == 2

        # neighbors 0
        assert_equal_vector(matrix[2, :], result.rows[0])
        assert vector_cos_dist(matrix[0, :], result.rows[0]) == result.distances[0].cpu()

        # neighbors 1
        assert equals(matrix[1, :], result.rows[1]) \
               or \
               equals(matrix[3, :], result.rows[1])
 
        assert (vector_cos_dist(matrix[0, :], result.rows[1]) == result.distances[1].cpu()) \
                or \
               (vector_cos_dist(matrix[0, :], result.rows[3]) == result.distances[1].cpu())

