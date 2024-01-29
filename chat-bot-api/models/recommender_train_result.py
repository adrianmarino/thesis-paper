from .model import Model
import typing


class RecommenderTrainResult(Model):
  metadata: typing.Dict[str, typing.Any] = {}

  class Config:
      arbitrary_types_allowed = True