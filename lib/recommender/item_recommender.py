from abc import ABC, abstractmethod


class ItemRecommender(ABC):
    @abstractmethod
    def recommend(self, item_id: int = None, k: int = 5):
        pass

    @property
    def name(self):
        pass