from torch.utils.data import Dataset
import pandas as pd
import pytorch_common.util as pu
import data as dt
from .movie_leans_tmdb_dataset import MovieLensTMDbDataset


def as_type(df, columns, dtype):
    for col in columns:
        df[col] = df[col].astype(dtype)
    return df


class MovieLensTMDBDatasetFactory:
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

        dataset = clazz.preprocessing(path, movies, interactions, filter_fn)

        del movies
        del interactions

        return MovieLensTMDbDataset(
            dataset          = dataset,
            transform        = transform,
            target_transform = target_transform,
            device           = device
        )

    def preprocessing(path, movies, interactions, filter_fn):
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