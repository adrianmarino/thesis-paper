import pandas as pd
from bunch import Bunch
import util as ut
from recommender import UserItemRecommender
from .result import UserSimilarItemRecommenderResult
from ..sim_items_mixin import SimItemsMixin



class UserSimilarItemRecommender(UserItemRecommender, SimItemsMixin):

    def __init__(
            self,
            collection_repository,
            dataset_repository,
            n_top_rated_user_items=20,
            n_sim_items=3
    ):
        self._collection_repository = collection_repository
        self._dataset_repository    = dataset_repository
        self.n_top_rated_user_items = n_top_rated_user_items
        self.n_sim_items = n_sim_items


    def recommend(self, user_id: int, k: int = 5):
        top_user_items = self._dataset_repository.find_top_rated_item_by_user_id(
            user_id,
            self.n_top_rated_user_items
        )

        recommendations = self._similar_items(item_ids=top_user_items.index.unique())

        return UserSimilarItemRecommenderResult(
            self.name,
            recommendations,
            k
        )


    @property
    def name(self):
        return self._collection_repository.collection.name