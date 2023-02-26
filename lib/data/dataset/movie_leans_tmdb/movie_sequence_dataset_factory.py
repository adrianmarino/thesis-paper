from ..rec_sys_dataset import RecSysDataset
from .sequence_dataset import SequenceDataset
import data as dt
import util as ut
import numpy as np
import logging
import torch
import pandas as pd


class MovieSequenceDatasetFactory:
    @classmethod
    def create(
        clazz,
        dataset,
        max_movie_seq_len : int = 15,
        feat_seq_len      : int = 5,
        id_seq_init       : int = 0,
        mask              = None,
        train_ds          = None,
        n_min             : int = 0
    ) -> SequenceDataset:
        assert feat_seq_len >= 2, 'feat_seq_len must be greater that or equal 2'

        df = dataset.drop(columns=['movie_seq']) if 'movie_seq' in dataset.columns else dataset

        if train_ds:
            df['movie_seq'] = df['movie_id'].apply(lambda id: int(train_ds.seq_by_id[id]) if id in train_ds.seq_by_id else None)
            df = df.dropna()
        else:
            df = dt.Sequencer(
                column       = 'movie_id',
                seq_col_name = 'movie_seq',
                init         = id_seq_init
            ).perform(df)

        sequences = clazz._sequences(df, max_movie_seq_len)

        logging.info(f'chunk movie sequences with size={feat_seq_len}')

        features, targets = dt.sequences_to_feats_target(
            sequences,
            feat_seq_len  = feat_seq_len,
            mask          = mask
        )

        features, targets = clazz._filter_complete_padded(features, targets)
        features, targets = clazz._filter_when_not_found_in_train(features, targets, train_ds)
        features, targets = clazz._filter_min_examples_by_target(features, targets, n_min=n_min)

        features = torch.from_numpy(features).to(dtype=torch.int32)
        targets  = torch.from_numpy(targets).to(dtype=torch.int32)

        return SequenceDataset(
            features,
            targets,
            id_by_seq = ut.df_to_dict(df, key='movie_seq', value='movie_id'),
            seq_by_id = ut.df_to_dict(df, key='movie_id', value='movie_seq')
        )


    @staticmethod
    def _filter_complete_padded(features, targets):
        with dt.progress_bar(len(features), 'Filter complete padded') as bar:
            filtered_features = []
            filtered_targets  = []
            for f, t in zip(features, targets):
                if sum(f) > 0:
                    filtered_features.append(f)
                    filtered_targets.append(t)
                else:
                    logging.info(f'Sequece completed padded: {f}')
                bar.update()
            return np.array(filtered_features), np.array(filtered_targets)


    @staticmethod
    def _filter_when_not_found_in_train(features, targets, train_ds):
        if train_ds == None:
            return np.array(features), np.array(targets)

        with dt.progress_bar(len(features), 'Filter target that not in train') as bar:
            filtered_features = []
            filtered_targets  = []
            n_excluded = 0 
            for f, t in zip(features, targets):
                if t in train_ds.targets:
                    filtered_features.append(f)
                    filtered_targets.append(t)
                else:
                    n_excluded += 1
                bar.update()

            logging.info(f'targets not in train set: {n_excluded}')
            return np.array(filtered_features), np.array(filtered_targets)


    @staticmethod
    def _filter_min_examples_by_target(features, targets, n_min):
        if n_min <= 0:
            return np.array(features), np.array(targets)

        data = pd.DataFrame(targets, columns=['target'])
        groups = data['target'].value_counts().reset_index()
        groups = groups[groups['target'] >= n_min]
        valid_targets = groups.index.values

        with dt.progress_bar(len(features), f'Filter target with more han {n_min} examples') as bar:
            filtered_features = []
            filtered_targets  = []
            n_excluded = 0 
            for f, t in zip(features, targets):
                if t in valid_targets:
                    filtered_features.append(f)
                    filtered_targets.append(t)
                else:
                    n_excluded += 1
                bar.update()

            logging.info(f'Excluded: {n_excluded}')
            return np.array(filtered_features), np.array(filtered_targets)


    @staticmethod
    def _sequences(df, max_movie_seq_len):
        logging.info('group movies by user id')
        grouped = df.groupby(['user_id'])

        logging.info(f'get ordered user movie sequences <= {max_movie_seq_len}')
        
        movie_seq_by_user = {}
        with dt.progress_bar(df['user_id'].shape[0], 'Get user movie sequences') as bar:
            for user_id in df['user_id']:
                df = grouped.get_group(user_id)

                if df.shape[0] <= max_movie_seq_len:
                    df = df.sort_values(['timestamp'])
                    movie_seq_by_user[user_id] = df['movie_seq'].values

                bar.update()

        return [movie_seq_by_user[user_id] for user_id in movie_seq_by_user.keys()]
