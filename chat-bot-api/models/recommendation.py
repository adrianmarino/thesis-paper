from .model import Model


class Recommendation(Model):
  title: str
  release: str
  description: str
  rating: float
  votes: list[str]