from pydantic import BaseModel, Field
from .recommendation import Recommendation
import typing

class ChatBotResultModel(BaseModel):
  content: str = Field(..., description="The raw textual content returned by the LLM.", examples=["1. The Matrix (1999) - A masterpiece..."])
  metadata: typing.Dict[str, typing.Any] = Field(default_factory=dict, description="Internal LLM metadata (e.g. prompts, prompt templates).")

  class Config:
    from_attributes = True

class RecommendationsMetadata(BaseModel):
  excluded_items: list[Recommendation] = Field(default_factory=list, description="A list of candidate Recommendation objects that were discarded by the filtering/ranking logic.")
  response: ChatBotResultModel | None = Field(None, description="The raw response object from the LLM, including the internal prompt used and the raw content generated.")
  elapsed_time: str | None = Field(None, description="The total execution time of the recommendation process.", examples=["3.45s"])
  logs: list[str] = Field(default_factory=list, description="Internal execution logs showing the details of the steps taken.")

  def __getitem__(self, item):
    return getattr(self, item)

  def __setitem__(self, key, value):
    setattr(self, key, value)

  def __contains__(self, item):
    return hasattr(self, item)

class Recommendations(BaseModel):
  items: list[Recommendation] = Field(..., description="The final list of recommended movies, retrieved via RAG/CF and filtered/ranked by the LLM.")
  metadata: RecommendationsMetadata | None = Field(None, description="Optional execution metadata populated if `include_metadata` is True.")

  @property
  def content(self): 
    if self.metadata and self.metadata.response:
      return self.metadata.response.content
    return ""

  @property
  def empty(self): return len(self.items) == 0

  @property
  def plain(self):
    if self.metadata and self.metadata.response:
      content = 'Prompt:\n-------\n' + self.metadata.response.metadata['prompt']
      content += '\n\nResponse:\n---------\n' + self.content
    else:
      content = 'Must include metadata to see a plant result!'
    return content

  class Config:
      arbitrary_types_allowed = True
