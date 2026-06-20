from pydantic import BaseModel, Field
from .cf_settings import CFSettings
from .rag_settings import RagSettings


class RecommendationSettings(BaseModel):
    llm                     : str  = Field('deepseek-r1:8b', description="Distribution or version of the language model to use for the initial generation of candidates (e.g., llama3, deepseek-r1:8b).")
    retry                   : int  = Field(2, description="Number of retries in case the LLM fails or returns a malformed response.")
    base_url                : str  = Field("", description="Base URL of the request (dynamically assigned at runtime).")
    plain                   : bool = Field(False, description="If True, the API will respond with plain text instead of structured JSON.")
    include_metadata        : bool = Field(False, description="If True, includes additional metadata in the response about the source of the recommendation (similarities, etc.).")
    rag                     : RagSettings = Field(default_factory=RagSettings)
    collaborative_filtering : CFSettings = Field(default_factory=CFSettings)

    class Config:
        arbitrary_types_allowed = True
