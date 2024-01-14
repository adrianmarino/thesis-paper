from .model import Model
import typing
from .user_message import UserMessage
from .recommender_settings import RecommenderSettings


class RecommendationsRequest(Model):
  message: UserMessage
  recommender_settings: RecommenderSettings

  class Config:
      arbitrary_types_allowed = True