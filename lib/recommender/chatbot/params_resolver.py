from abc import ABC, abstractmethod


class ParamsResolver(ABC):
    @abstractmethod
    def __call__(self, **kargs):
        pass