import pandas as pd
from bunch import Bunch

import util as ut
from recommender import ItemRecommender
from .result import ItemEmbDBRecommenderResult
from ..sim_items_mixin import SimItemsMixin


def group_mean(df, group_col, mean_col):
    return df.groupby([group_col])[mean_col].mean().reset_index()


def mean_by_key(df, key, value):
    ut.to_dict(group_mean(df, key, value), key, value)


class ItemEmbDBRecommender(ItemRecommender, SimItemsMixin):

    def __init__(
            self,
            collection_repository,
            dataset_repository,
            n_sim_items=3
    ):
        self._collection_repository = collection_repository
        self._dataset_repository = dataset_repository
        self.n_sim_items = n_sim_items


    def recommend(self, item_id: int, k: int = 5):
        recommendations = self._similar_items([item_id])

        sorted_recommendations = recommendations.sort_values(by=['rating'], ascending=False)

        return ItemEmbDBRecommenderResult(sorted_recommendations.head(k))