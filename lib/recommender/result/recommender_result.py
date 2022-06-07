from abc import ABC, abstractmethod


class RecommenderResult(ABC):
    @abstractmethod
    def show(self):
        pass