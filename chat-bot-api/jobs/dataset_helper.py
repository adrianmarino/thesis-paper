import pandas as pd
import torch
import data as dt
import data.dataset as ds
import logging
import rest


def load_raw_dataset(path, start_year=2004):
    def to_tensor(obs, device, columns):
        data = obs[columns]
        if type(data) == pd.DataFrame:
            data = data.values
        return torch.tensor(data).to(device)

    features_fn = lambda obs, device: to_tensor(obs, device, ['user_seq', 'movie_seq'])
    target_fn   = lambda obs, device: to_tensor(obs, device, ['user_movie_rating'])

    return ds.MovieLensTMDBDatasetFactory.from_path(
        path             = path,
        transform        = features_fn,
        target_transform = target_fn,
        device           = torch.device('cpu'),
        filter_fn        = lambda df: df[(df['user_movie_rating_year'] >= start_year)]
    )


def train_test_split_and_filer_cols(dataset, split_year=2016):
    train_set, eval_test_set, _, _ = dataset.train_test_split(split_year=split_year)

    train_set = train_set \
        .data[['movie_id', 'movie_seq', 'user_id', 'user_seq', 'user_movie_rating']] \
        .rename(columns={
            'movie_id'          : 'item_id',
            'movie_seq'         : 'item_seq',
            'user_movie_rating' : 'rating'
        })

    eval_test_set = eval_test_set \
        .data[['movie_id', 'movie_seq', 'user_id', 'user_seq', 'user_movie_rating']] \
        .rename(columns={
            'movie_id'          : 'item_id',
            'movie_seq'         : 'item_seq',
            'user_movie_rating' : 'rating'
        })

    return train_set, eval_test_set


def to_interactions_set(ds, interactions_df):
    logging.info(f'used_ids: {len(ds["user_id"].unique())}, item_ids: {len(ds["item_id"].unique())}.')
    logging.info(f'Max used_seq: {ds["user_seq"].max()}, Max item_seq: {ds["item_seq"].max()}.')

    sequencer       = dt.Sequencer(
        column       = 'user_id',
        seq_col_name = 'user_seq',
        offset       = ds['user_seq'].max() + 1
    )
    interactions_df = sequencer.perform(interactions_df)

    interactions_df['item_id'] = interactions_df['item_id'].astype(int)

    item_id_seq_df = ds[['item_id', 'item_seq']].drop_duplicates(subset=['item_id'])

    interactions_df = interactions_df.merge(item_id_seq_df, on=['item_id'])

    return interactions_df[ds.columns]


def build_datasets(
    path,
    interactions_df,
    start_year    = 2004,
    split_year    = 2018,
    int_test_size = 0.1
):
    dataset = load_raw_dataset(path, start_year)

    dev_set, test_set = train_test_split_and_filer_cols(dataset, split_year)

    interactions_set = to_interactions_set(dev_set, interactions_df)

    if len(interactions_df) > 200:
        logging.info('(interactions > 200) => Split interactions and add to dev_set and test_set.')
        int_dev_set, int_test_set = train_test_split(
            interactions_set,
            test_size = int_test_size,
            shuffle   = True
        )
        dev_set  = pd.concat([int_dev_set,  dev_set ])
        test_set = pd.concat([int_test_set, test_set])
    else:
        logging.info('(interactions < 200) => Add interactions to dev_set.')
        dev_set  = pd.concat([interactions_set, dev_set])

    return dev_set, test_set


def get_interactions():
    return pd.DataFrame(rest.ChatBotV1ApiClient().interactions())