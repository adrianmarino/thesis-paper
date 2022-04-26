import logging
import data.plot as pl
import pytorch_common.util as pu
import torch


def rows_distance_matrix(matrix, distance_fn, device=pu.get_device()):
    sw = pu.Stopwatch()
    distances = torch.zeros([matrix.shape[0]]*2).to(device)

    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx, :]
        distances[row_idx, :] = distance_fn(row, matrix[:, :])

    logging.info(f'distances matrix - computing time: {sw.to_str()}')
    return distances