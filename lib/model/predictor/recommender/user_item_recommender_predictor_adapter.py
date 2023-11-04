from model.predictor.abstract_predictor import AbstractPredictor
import recommender as rc
import logging
import util as ut
import torch

class UserItemRecommenderPredictorAdapter(AbstractPredictor):
    def __init__(self, recommender: rc.UserItemRecommender):
        self.recommender      = recommender
        self.items_by_user_id = {}
        self.ratings_cache = {}

    @property
    def name(self): return self.recommender.name


    def _get_items_by(self, user_idx):
        if user_idx in self.items_by_user_id:
            return self.items_by_user_id[user_idx]
        else:
            result = self.recommender.recommend(user_idx, k=None).data

            self.items_by_user_id[user_idx] = result
            return result


    def predict(self, user_idx, item_idx, n_neighbors=10, debug=False):
        key = f'{user_idx}-{item_idx}'
        if key in self.ratings_cache:
            return self.ratings_cache[key]
        else:
            items_df = self._get_items_by(user_idx)

            items_df = items_df.query(f'{self.recommender.item_id_col} == {item_idx}')

            rating = items_df[self.recommender.rating_col].values[0]

            self.ratings_cache[key] = rating

            return rating


    def predict_batch(self, batch, n_neighbors=10, debug=False):
        results = ut.ParallelExecutor()(
            self.predict,
            params = [[batch[idx][0], batch[idx][1],  n_neighbors, debug] for idx in range(len(batch))],
            fallback_result = 0
        )

        return torch.tensor(results)
