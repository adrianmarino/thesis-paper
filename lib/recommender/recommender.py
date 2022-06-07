from abc import ABC, abstractmethod


class Recommender(ABC):
    @abstractmethod
    def recommend(self, item_index: int, user_id: int=None, k: int=5):
        pass
