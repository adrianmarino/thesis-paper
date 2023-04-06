from torch.utils.data import Dataset
import numpy as np
from sklearn import preprocessing


def build_user_rating_emb(movie_seqs, user_ratings, emb_size):
    emb = np.zeros(emb_size)
    for seq, rating in zip(movie_seqs, user_ratings):
        emb[seq] = int(rating)
    return emb


def build_user_rating_emb_fn(emb_size, column):
    return lambda row: build_user_rating_emb(
        row['movie_seq'],
        row[column],
        emb_size + 1
    )


def standard_scale(values):
    standardizer = preprocessing.StandardScaler()
    values = standardizer.fit_transform(values.reshape(-1, 1))
    return values, standardizer


def standard_scale_col(df, source, target):
    values, standardizer = standard_scale(df[source].values)
    df[target] = values
    return standardizer


def data_preprocessing(df):
    standardizer = standard_scale_col(
        df,
        'user_movie_rating',
        'rating_scaled'
    )

    # Aggregate movie seq y ratings by user...
    user_movies = df \
        .sort_values(['user_seq', 'user_movie_rating_timestamp']) \
        .groupby(by=['user_seq']) \
        .agg({
            'movie_seq':lambda x: list(x),
            'rating_scaled':lambda x: list(x)
        }) \
        .reset_index()

    user_movies['movie_rating_emb'] = user_movies \
        .apply(
            build_user_rating_emb_fn(
                emb_size = df.movie_seq.max(),
                column   = 'rating_scaled'
            ),
            axis=1
        )

    user_movies['user_seq'] = user_movies['user_seq'].astype('int32')
    user_movies['movie_rating_emb'] = user_movies['movie_rating_emb'].apply(lambda x: np.float16(x))

    return user_movies, standardizer


class CollaborativeVariationalAutoEncoderDataset(Dataset):
    def __init__(self, df):
        self.data, self.standardizer = data_preprocessing(df)
        self.raw_data = df


    def __len__(self)              : return self.data.shape[0]


    def __getitem__(self, row_idx):
        """
        returns a tuple (features, target), Where:
        - features = [ (user_movie_ratings,) ]
        - target   = [ (user_movie_ratings,) ]              ]
        """

        data_set = self.data.iloc[row_idx, :]

        ratings = data_set['movie_rating_emb'].astype(np.float32)

        return ratings, ratings