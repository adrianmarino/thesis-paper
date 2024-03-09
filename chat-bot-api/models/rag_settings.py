from .model import Model
from .recommender_settings import RecommenderSettings
import typing


class RagSettings(RecommenderSettings):
    not_seen : bool  = True