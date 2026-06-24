from .recommender_settings import RecommenderSettings
from pydantic import Field

class RagSettings(RecommenderSettings):
    not_seen : bool  = Field(True, description="If True, the RAG process will filter out movies that the user has already seen.")
    min_rating_by_user : float = Field(0.0, description="Minimum rating required for a candidate movie in RAG.")
