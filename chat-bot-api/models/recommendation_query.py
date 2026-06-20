from pydantic import BaseModel, Field
from .user_message import UserMessage
from .recommendation_settings import RecommendationSettings


class RecommendationQuery(BaseModel):
    message: UserMessage = Field(..., description="The user's message and identity.")
    settings: RecommendationSettings = Field(default_factory=RecommendationSettings, description="Advanced hyperparameter configurations for the hybrid recommendation engine.")

    class Config:
        arbitrary_types_allowed = True
