import numpy as np
import sys
import pandas as pd
import service as srv
import logging
import sys
sys.path.append(ctx['recsys.client.src_path'])
sys.path.append(ctx['thesis.src_path'])


class ModuleComputeRatingMatrixTask:
    def __init__(
        self,
        task_id,
        model_loader,
        domain,
    ):
        self.task_id  = task_id
        self.domain   = domain
        self.service  = srv.ModulePredictionService(model_loader)

    def __save_interactions(self, df, name):
        df.to_json(
            f'{self.domain.cfg.temp_path}/{self.task_id}_{name}_interactions.json',
            orient='records'
        )

    def __load(self, path):
        return pd.read_json(f'{self.domain.cfg.temp_path}/{path}', orient='records')

    def __train_predict(self, train_df, test_df, columns):


    def perform(
        self,
        interactions_path,
        min_n_interactions = 20,
        rating_scale       = np.arange(0, 6, 0.5),
        columns            = ('user_seq', 'item_seq', 'rating')
    ):
        interactions = self.__load(interactions_path)

        logging.info('HOLLIII')

        # Build ratings matrix from user-item interactions..
        future_interactions, filtered_train_interactions = self.domain.interaction_inference_service.predict(
            train_interactions = interactions,
            columns            = columns,
            train_predict_fn   = lambda train_df, test_df, _: self.service.predict(train_df, test_df),
            min_n_interactions = min_n_interactions,
            rating_scale       = rating_scale
        )

        future_interactions = future_interactions \
            .drop(columns=['rating']) \
            .rename(columns={'rating_prediction': 'rating'}) \
            .query('rating > 0')

        self.__save_interactions(future_interactions, 'future')

        self.__save_interactions(filtered_train_interactions, 'train')
