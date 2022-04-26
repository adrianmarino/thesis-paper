from abc import ABC, abstractmethod


class AbstractPredictor(ABC):
    def __init__(self, name=None): self._name = name

    @abstractmethod    
    def predict(self, user_idx, item_idx, debug=False):
        pass