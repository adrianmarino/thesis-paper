import pytorch_common.util as pu

from .movie_leans_tmdb_dataset import MovieLensTMDbDataset
from .movie_lens_tmdb_data_loader import MovieLensTMDBDataLoader
import torch


def as_type(df, columns, dtype):
    for col in columns:
        df[col] = df[col].astype(dtype)
    return df


class MovieLensTMDBDatasetFactory:
    @classmethod
    def from_path(
            cls,
            path             = '../datasets',
            transform        = lambda obs, device: obs[['user_seq', 'movie_seq']].values,
            target_transform = lambda obs, device: obs['user_movie_rating'],
            device           = torch.device('cpu'),
            filter_fn        = lambda df: df
    ):
        return MovieLensTMDbDataset(
            dataset=MovieLensTMDBDataLoader.df_from_path(path, filter_fn),
            transform=transform,
            target_transform=target_transform,
            device=device
        )
