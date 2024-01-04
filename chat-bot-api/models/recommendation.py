from .model import Model
import typing


class Recommendation(Model):
  title: str
  release: str
  description: str
  votes: list[str]
  metadata: typing.Dict[str, typing.Any] | None