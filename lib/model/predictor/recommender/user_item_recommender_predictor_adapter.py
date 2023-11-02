from model.predictor.abstract_predictor import AbstractPredictor
import recommender as rc


class UserItemRecommenderPredictorAdapter(AbstractPredictor):
    def __init__(self, recommender: rc.UserItemRecommender):
        self.recommender      = recommender
        self.items_by_user_id = {}


    @property
    def name(self): return self.recommender.name


    def _get_items_by(self, user_idx):
        if user_idx in self.items_by_user_id:
            return self.items_by_user_id[user_idx]
        else:
            result = self.recommender.recommend(user_idx, k=None).data

            self.items_by_user_id[user_idx] = result
            return result


    def predict(self, user_idx, item_idx):
        items_df = self._get_items_by(user_idx)

        items_df = items_df.query(f'{self.recommender.item_id_col} == {item_idx}')

        return items_df[self.recommender.rating_col].values[0]