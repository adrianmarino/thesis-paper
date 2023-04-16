from ..rec_sys_dataset import RecSysDataset
import pandas as pd
import logging


class MovieLensTMDbDataset(RecSysDataset):
    def __init__(self, dataset, transform, target_transform, device):
        super().__init__(dataset, transform, target_transform, device)

    def train_test_split(self, split_year=2018, rating_mean_df=pd.DataFrame(), rating_std=None):
        # ----------
        # Train set:
        # ----------
        df_train = self.data[self.data['user_movie_rating_year'] < split_year]
        df_train = df_train.loc[:, ~df_train.columns.isin(['user_movie_rating_mean', 'user_movie_rating_norm'])]

        # Add mean rating column by user...
        if rating_mean_df.empty:
            rating_mean_df = df_train \
                .groupby('user_seq')['user_movie_rating'] \
                .mean() \
                .reset_index(name='user_movie_rating_mean')

        df_train = pd.merge(
            df_train,
            rating_mean_df,
            how='inner',
            left_on=['user_seq'],
            right_on=['user_seq']
        ).dropna()

        # Users rating std deviation.......
        if rating_std == None:
            rating_std = df_train['user_movie_rating'].std()


        # Create normalized rattings column...
        df_train['user_movie_rating_norm'] = df_train \
            .apply(lambda row: round((row['user_movie_rating'] - row['user_movie_rating_mean']) / rating_std, 2), axis=1)

        # ---------------
        # Validation set:
        # ---------------
        # - Include only movies an used that exists in train set.
        df_eval  = self.data[
            (self.data['user_movie_rating_year'] >= split_year) &
            (self.data['user_seq' ].isin(df_train.user_seq.values)) &
            (self.data['movie_seq'].isin(df_train.movie_seq.values))
        ]
        df_eval = df_eval.loc[:, ~df_eval.columns.isin(['user_movie_rating_mean', 'user_movie_rating_norm'])]

        df_eval = pd.merge(
            df_eval,
            rating_mean_df,
            how='inner',
            left_on=['user_seq'],
            right_on=['user_seq']
        ).dropna()

        # Create normalized rattings column...
        df_eval['user_movie_rating_norm'] = df_eval \
            .apply(lambda row: round((row['user_movie_rating']-row['user_movie_rating_mean']) / rating_std, 2), axis=1)


        # Create a dataset for each set...
        train_data    = self.__subset(df_train)
        test_data     = self.__subset(df_eval)

        logging.info(f'Train: {(train_data.shape[0]/len(self))*100:.2f} % - Test: {(test_data.shape[0]/len(self))*100:.2f} %')

        return train_data, test_data, rating_mean_df, rating_std

    def __subset(self, data):
        return MovieLensTMDbDataset(
            dataset          = data,
            transform        = self._transform[0],
            target_transform = self._target_transform,
            device           = self._device
        )
