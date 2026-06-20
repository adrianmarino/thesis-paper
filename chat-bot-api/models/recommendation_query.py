from pydantic import BaseModel, Field
from .user_message import UserMessage
from .recommendation_settings import RecommendationSettings


class RecommendationQuery(BaseModel):
    message: UserMessage = Field(..., description="El mensaje del usuario y su identidad.")
    settings: RecommendationSettings = Field(default_factory=RecommendationSettings, description="Configuraciones avanzadas de hiperparámetros para el motor de recomendación híbrido.")

    class Config:
        arbitrary_types_allowed = True
