import pandas as pd
from bunch import Bunch
import util as ut
from recommender import ItemRecommender
from .result import SimilarItemRecommenderResult
from ..sim_items_mixin import SimItemsMixin


class SimilarItemRecommender(ItemRecommender, SimItemsMixin):

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
        return SimilarItemRecommenderResult(
            self.name,
            self._similar_items([item_id]),
            k
        )

    @property
    def name(self):
        return self._collection_repository.collection.name