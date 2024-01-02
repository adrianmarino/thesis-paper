from .model import Model


class Recommendation(Model):
  title: str
  release: str
  description: str
  rating: float
  votes: list[str]
  total_sim: float
  db_title_sim: float
  db_title: str