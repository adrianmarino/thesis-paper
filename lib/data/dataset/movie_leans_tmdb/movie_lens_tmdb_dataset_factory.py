import pytorch_common.util as pu

from .movie_leans_tmdb_dataset import MovieLensTMDbDataset
from .movie_lens_tmdb_data_loader import MovieLensTMDBDataLoader


def as_type(df, columns, dtype):
    for col in columns:
        df[col] = df[col].astype(dtype)
    return df


class MovieLensTMDBDatasetFactory:
    @classmethod
    def from_path(
            cls,
            transform,
            target_transform,
            path='../datasets',
            device=pu.get_device(),
            filter_fn=lambda df: df
    ):
        return MovieLensTMDbDataset(
            dataset=MovieLensTMDBDataLoader.df_from_path(path, filter_fn),
            transform=transform,
            target_transform=target_transform,
            device=device
        )
