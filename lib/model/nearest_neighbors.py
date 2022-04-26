import torch
import pytorch_common.util as pu
import model as ml


class NearestNeighborsResult:
    def __init__(self, rows, distances, indexes):
        self.rows = rows
        self.distances = distances
        self.indexes = indexes


class NearestNeighbors:
    def __init__(self, matrix, distance_fn, device=pu.get_device()):
        self.matrix = matrix
        self.row_distances = ml.rows_distance_matrix(matrix, distance_fn, device)

    def _k_nearest_row_indexes(self, row_idx, k):
        row = self.row_distances[row_idx, :]
        orderd_distances, ordered_indexes = torch.sort(row, descending=False)

        if k >= self.matrix.shape[0]: k = self.matrix.shape[0] -1

        return orderd_distances[1:k+1], ordered_indexes[1:k+1]

    def neighbors(self, row_idx, k):
        k_nearest_row_distances, k_nearest_row_indexes = self._k_nearest_row_indexes(row_idx, k)

        return NearestNeighborsResult(
            rows      = self.matrix[k_nearest_row_indexes, :], 
            distances = k_nearest_row_distances, 
            indexes   = k_nearest_row_indexes
        )
