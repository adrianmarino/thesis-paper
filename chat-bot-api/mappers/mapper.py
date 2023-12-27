from abc import ABC, abstractmethod


class ModelMapper:
  @abstractmethod
  def to_model(self, document):
    pass

  @abstractmethod
  def to_dict(self, model):
    pass


