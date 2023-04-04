from torch.utils.data import Dataset
import numpy as np


def build_user_rating_emb(movie_seqs, user_ratings, emb_size):
    emb = np.zeros(emb_size)
    for seq, rating in zip(movie_seqs, user_ratings):
        emb[seq] = int(rating)
    return emb


def build_user_rating_emb_fn(emb_size):
    return lambda row: build_user_rating_emb(
        row['movie_seq'],
        row['user_movie_rating'],
        emb_size + 1
    )


def data_preprocessing(df):
    user_rating_emb_size = df.movie_seq.max()
    user_rating_emb_size

    user_movies = df \
        .sort_values(['user_seq', 'user_movie_rating_timestamp']) \
        .groupby(by=['user_seq']) \
        .agg({
            'movie_seq':lambda x: list(x),
            'user_movie_rating':lambda x: list(x)
        }) \
        .reset_index()

    user_movies['movie_rating_emb'] = user_movies \
        .apply(build_user_rating_emb_fn(user_rating_emb_size), axis=1)

    result = user_movies \
        .drop(columns=['movie_seq', 'user_movie_rating'])

    result['user_seq'] = result['user_seq'].astype('int32')
    result['movie_rating_emb'] = result['movie_rating_emb'].apply(lambda x: np.float16(x))

    return result


class CollaborativeVariationalAutoEncoderDataset(Dataset):
    def __init__(self, df)         : self.data = data_preprocessing(df)


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