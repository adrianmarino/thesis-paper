from pydantic import BaseModel
from .user_message import UserMessage
from .recommendation_settings import RecommendationSettings


class RecommendationQuery(BaseModel):
    message: UserMessage
    settings: RecommendationSettings

    class Config:
        arbitrary_types_allowed = True
