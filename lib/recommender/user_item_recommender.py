from abc import ABC, abstractmethod


class UserItemRecommender(ABC):
    @abstractmethod
    def recommend(self, user_id: int = None, k: int = 5):
        pass

    @property
    def name(self):
        pass