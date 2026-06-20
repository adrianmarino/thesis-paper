from .recommender_settings import RecommenderSettings
from pydantic import Field

class RagSettings(RecommenderSettings):
    not_seen : bool  = Field(True, description="Si es True, el proceso de RAG filtrará las películas que el usuario ya haya visto.")
