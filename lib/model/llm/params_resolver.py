from abc import ABC, abstractmethod


class ParamsResolver(ABC):
    @abstractmethod
    def resolve(self, **kargs):
        pass