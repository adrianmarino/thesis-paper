import torch
from torch.utils.data import Dataset
import pandas as pd
import data as dtypes
import pytorch_common.util as pu
import data as dt
import numpy as np
import logging


def as_type(df, columns, dtype):
    for col in columns:
        df[col] = df[col].astype(dtype)
    return df


class MovieLensTMDbJSONDataset(Dataset):
    @classmethod
    def from_path(
        clazz,
        transform,
        target_transform,
        path      = '../datasets',
        device    = pu.get_device(),
        filter_fn = lambda df: df
    ):
        movies       = pd.read_json(f'{path}/movies.json')
        interactions = pd.read_json(f'{path}/interactions.json')

        dataset = clazz.__join_and_curate_data(path, movies, interactions, filter_fn)

        del movies
        del interactions

        return MovieLensTMDbJSONDataset(
            dataset          = dataset,
            transform        = transform,
            target_transform = target_transform,
            device           = device
        )

    def __init__(self, dataset, transform, target_transform, device):
        self.__dataset          = dataset
        self.__transform        = transform,
        self.__target_transform = target_transform
        self.__device           = device

    def __len__(self): return len(self.data)


    def split_train_eval(self, split_year=2018):
        df_train = self.data[self.data['user_movie_rating_year'] < split_year]

        df_eval  = self.data[
            (self.data['user_movie_rating_year'] >= split_year) & 
            (self.data['user_seq' ].isin(df_train.user_seq.values)) & 
            (self.data['movie_seq'].isin(df_train.movie_seq.values))
        ]

        train_data    = self.subset_by_indexes(df_train.index.tolist())
        test_data     = self.subset_by_indexes(df_eval.index.tolist())

        logging.info(f'Train: {(train_data.shape[0]/len(self))*100:.2f} % - Test: {(test_data.shape[0]/len(self))*100:.2f} %')

        return train_data, test_data

    
    def __getitem__(self, rows_idx):
        try:
            return self.__to_feats_target(self.data.iloc[rows_idx, :])
        except Exception as e:
            logging.error(f'Error to index dataset with: {rows_idx}. dataset size: {len(self.data)}')
            raise e


    def sample(self, size):
        indexes = torch.randint(0, len(self)-1, (size,))
        df_sample = pd.DataFrame(self.data.iloc[indexes, :])
        features, target = self.__to_feats_target(df_sample)
        return features, target


    def subset_by_indexes(self, indexes):
        return MovieLensTMDbJSONDataset(
            dataset          = self.data.iloc[indexes, :],
            transform        = self.__transform[0],
            target_transform = self.__target_transform,
            device           = self.__device
        )


    @property
    def data(self): return self.__dataset


    @property
    def columns(self): return self.data.columns


    @property
    def dtypes(self): return self.data.dtypes


    @property
    def info(self): return self.data.info()


    @property
    def shape(self): return self.data.shape


    @property
    def features(self): return self.__to_feats(self.data)

        
    @property
    def targets(self): return self.__to_target(self.data)


    @property
    def features_uniques(self):         
        return [ c[0] for c in self.features_value_counts]


    @property
    def target_uniques(self): return self.targets_value_counts[0]


    @property
    def features_value_counts(self):
        features = self.features
        return np.array([ np.unique(features[:, col_idx].cpu().numpy(), return_counts=True) for col_idx in range(features.shape[1])], dtype=object)


    @property
    def targets_value_counts(self): 
        return np.unique(self.targets.numpy(), return_counts=True)

    def __to_feats_target(self, observations):
        features    = self.__to_feats(observations)
        target      = self.__to_target(observations)

        return features, target


    def __to_feats(self, observations):
        return self.__transform[0](observations, self.__device)


    def __to_target(self, observations):
        return self.__target_transform(observations, self.__device)


    @classmethod
    def __join_and_curate_data(clazz, path, movies, interactions, filter_fn):
        # Renaming...
        m = movies.rename(columns={c:f'movie_{c}' for c in  movies.columns})
        i = interactions.rename(columns={
            'movie_id': 'inter_movie_id',
            'rating': 'user_movie_rating',
            'timestamp': 'user_movie_rating_timestamp',
            'tags': 'user_movie_tags'
        })


        # join tables...
        dataset = pd.merge(
            m,
            i,
            how='inner',
            left_on=['movie_id'],
            right_on=['inter_movie_id'],
        )


        # Exclude columns...
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


        # Set data types...
        dataset = as_type(
            dataset,
            columns = [
                'movie_title', 
                'movie_original_language', 
                'movie_overview'
            ],
            dtype = 'string'
        )
        dataset = as_type(
            dataset,
            columns = ['movie_for_adults'],
            dtype = 'bool'
        )


        # Ratings...
        rattings_mapping = {
            0.5: 1,
            1.0: 1,
            1.5: 1,
            2.0: 2,
            2.5: 2,
            3.0: 3,
            3.5: 3,
            4.0: 4,
            4.5: 4,
            5.0: 5
        }
        dataset['user_movie_rating'] = \
            dataset['user_movie_rating'].apply(lambda it: rattings_mapping[it])


        # Years...
        dataset['user_movie_rating_year'] = \
            pd.DatetimeIndex(dataset['user_movie_rating_timestamp']).year


        # Filter dataset...
        dataset = filter_fn(dataset)


        # Generate sequences....
        user_seq = dt.Sequencer('user_id', 'user_seq')
        movie_seq = dt.Sequencer('movie_id', 'movie_seq')
        
        dataset = user_seq.perform(dataset)
        dataset = movie_seq.perform(dataset)

        dataset[['user_seq', 'user_id']] \
            .drop_duplicates() \
            .sort_values(by='user_seq', ascending=False) \
            .to_json(f'{path}/user_seq_id.json')
        
        dataset[['movie_seq', 'movie_id']] \
            .drop_duplicates() \
            .sort_values(by='movie_seq', ascending=False) \
            .to_json(f'{path}/movie_seq_id.json')

        return dataset[[
            'user_id',
            'user_seq',
            'user_movie_tags',
            'user_movie_rating',
            'user_movie_rating_timestamp',
            'user_movie_rating_year',
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