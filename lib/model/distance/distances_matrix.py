import torch
import pytorch_common.util as pu
from data.plot import headmap
import data as dt


def rows_distance_matrix(matrix, distance_fn, device=pu.get_device()):
    distances = torch.zeros([matrix.shape[0]]*2).to(device)

    with dt.progress_bar(matrix.shape[0], 'Building Distances Matrix') as bar:
        for index, row_idx in enumerate(range(matrix.shape[0])):
            row = matrix[row_idx, :]
            distances[row_idx, :] = distance_fn(row, matrix[:, :])
            bar.update()

    return distances


def plot_rows_distance_matrix(dm, figsize = (10, 10), prefix=''):
    dt.plot.headmap(
        dm.cpu(),
        title=f'{prefix}Distances Matrix ({dm.shape[0]},{dm.shape[1]})',
        figsize=figsize
    )
