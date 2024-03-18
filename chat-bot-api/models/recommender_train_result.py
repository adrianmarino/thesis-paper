from pydantic import BaseModel
import typing


class RecommenderTrainResult(BaseModel):
  metadata: typing.Dict[str, typing.Any] = {}

  class Config:
      arbitrary_types_allowed = True