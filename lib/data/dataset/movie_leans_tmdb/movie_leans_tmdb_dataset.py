from ..rec_sys_dataset import RecSysDataset
import pandas as pd
import logging


class MovieLensTMDbDataset(RecSysDataset):
    def __init__(self, dataset, transform, target_transform, device):
        super().__init__(dataset, transform, target_transform, device)

    def split_train_eval(self, split_year=2018):
        # ----------
        # Train set:
        # ----------
        df_train = self.data[self.data['user_movie_rating_year'] < split_year]

        # Add mean rating column by user...
        user_mean_rating = df_train \
            .groupby('user_seq')['user_movie_rating'] \
            .mean() \
            .reset_index(name='user_movie_rating_mean')

        df_train = pd.merge(
            df_train,
            user_mean_rating,
            how='inner',
            left_on=['user_seq'],
            right_on=['user_seq']
        )
    
        # Users rating std deviation.......
        train_user_rating_std = df_train['user_movie_rating'].std()
        
        # Create normalized rattings column...
        df_train['user_movie_rating_norm'] = df_train \
            .apply(lambda row: abs(row['user_movie_rating']-row['user_movie_rating_mean'])/train_user_rating_std, axis=1) 

        # ---------------
        # Validation set:
        # ---------------
        # - Include only movies an used that exists in train set.
        df_eval  = self.data[
            (self.data['user_movie_rating_year'] >= split_year) & 
            (self.data['user_seq' ].isin(df_train.user_seq.values)) & 
            (self.data['movie_seq'].isin(df_train.movie_seq.values))
        ]

        df_eval = pd.merge(
            df_eval,
            user_mean_rating,
            how='inner',
            left_on=['user_seq'],
            right_on=['user_seq']
        )

        # Create normalized rattings column...
        df_eval['user_movie_rating_norm'] = df_eval \
            .apply(lambda row: abs(row['user_movie_rating']-row['user_movie_rating_mean'])/train_user_rating_std, axis=1) 


        # Create a dataset for each set...
        train_data    = self.__subset(df_train)
        test_data     = self.__subset(df_eval)

        logging.info(f'Train: {(train_data.shape[0]/len(self))*100:.2f} % - Test: {(test_data.shape[0]/len(self))*100:.2f} %')

        return train_data, test_data

    def __subset(self, data):
        return MovieLensTMDbDataset(
            dataset          = data,
            transform        = self._transform[0],
            target_transform = self._target_transform,
            device           = self._device
        )
