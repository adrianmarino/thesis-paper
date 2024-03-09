from .model import Model
from .user_message import UserMessage
from .recommendation_settings import RecommendationSettings


class RecommendationQuery(Model):
    message: UserMessage
    settings: RecommendationSettings

    class Config:
        arbitrary_types_allowed = True
