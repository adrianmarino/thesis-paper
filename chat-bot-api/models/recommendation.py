from pydantic import BaseModel
import typing


class Recommendation(BaseModel):
  title: str
  poster: str | None
  release: str
  description: str
  genres: list[str]
  votes: list[str]
  metadata: typing.Dict[str, typing.Any] | None