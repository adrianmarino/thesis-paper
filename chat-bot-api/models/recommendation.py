from .model import Model
import typing


class Recommendation(Model):
  title: str
  poster: str | None
  release: str
  description: str
  rating: float
  genres: list[str]
  votes: list[str]
  metadata: typing.Dict[str, typing.Any] | None