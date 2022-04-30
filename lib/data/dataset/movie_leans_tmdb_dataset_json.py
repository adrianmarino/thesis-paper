import torch
from torch.utils.data import Dataset
import pandas as pd
import data as dtypes
import pytorch_common.util as pu
import data as dt


class MovieLensTMDbJSONDataset(Dataset):

    def __init__(
        self, 
        transform,
        target_transform,
        path='../datasets',
        device=pu.get_device()
    ): 
        self.__transform        = transform,
        self.__target_transform = target_transform
        self.__device           = device

        movies       = pd.read_json(f'{path}/movies.json')
        interactions = pd.read_json(f'{path}/interactions.json')

        self.__generate_sequences(path, interactions)
        self.__dataset = self.__join_and_curate_data(movies, interactions)

        del movies
        del interactions

    def __len__(self): return len(self.__dataset)

    def __getitem__(self, rows_idx):
        return self.__to_feats_labels(self.__dataset.iloc[rows_idx, :])

    @property
    def data(self): return self.__dataset

    @property
    def columns(self): return self.data.columns

    @property
    def dtypes(self): return self.data.dtypes

    def sample(self, size):
        indexes = torch.randint(0, len(self)-1, (size,))
        return self.__to_feats_labels(self.__dataset.iloc[indexes])

    def __to_feats_labels(self, observations):
        features    = self.__transform[0](observations, self.__device)
        label       = self.__target_transform(observations, self.__device)

        return features, label

    def __generate_sequences(self, path, interactions):
        interactions['user_seq']  = interactions.user_id.apply(dt.Sequencer().get)
        interactions['movie_seq'] =  interactions.movie_id.apply(dt.Sequencer().get)

        interactions[['user_seq', 'user_id']].to_json(f'{path}/user_seq_id.json')
        interactions[['movie_seq', 'movie_id']].to_json(f'{path}/movie_seq_id.json')

    def __join_and_curate_data(self, movies, interactions):
        m = movies.rename(columns={c:f'movie_{c}' for c in  movies.columns})

        i = interactions.rename(columns={
            'movie_id': 'inter_movie_id',
            'rating': 'user_movie_rating',
            'timestamp': 'user_movie_rating_timestamp',
            'tags': 'user_movie_tags'
        })

        dataset = pd.merge(
            m,
            i,
            how='inner',
            left_on=['movie_id'],
            right_on=['inter_movie_id'],
        )

        exclude_cols = lambda df, columns: df.loc[:, ~df.columns.isin(columns)]

        excluded = [
            'movie_poster', 
            'movie_popularity', 
            'inter_movie_id',
            'movie_vote_mean',
            'movie_vote_count',
            'movie_release',
            'year'
        ]

        dataset = dataset.pipe(exclude_cols, excluded)

        return dataset[[
            'user_id',
            'user_seq',
            'user_movie_tags',
            'user_movie_rating',
            'user_movie_rating_timestamp',
            'movie_id',
            'movie_seq',
            'movie_title',
            'movie_genres',
            'movie_for_adults',
            'movie_original_language',
            'movie_overview',
            'movie_tags',
            'movie_release_year'
        ]]