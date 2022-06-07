from .recommender import Recommender
from .result.impl.recommender_result_group import RecommenderResultGroup


class RecommenderGroup(Recommender):
    def __init__(self, recommenders): self.__recommenders = recommenders
    def recommend(self, item_index, user_id=None, k=5):
        return RecommenderResultGroup([r.recommend(item_index, user_id, k) for r in self.__recommenders])
