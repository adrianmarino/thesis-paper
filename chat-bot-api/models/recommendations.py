from pydantic import BaseModel, Field
from .recommendation import Recommendation
import typing

class Recommendations(BaseModel):
  items: list[Recommendation] = Field(..., description="The final list of recommended movies, retrieved via RAG/CF and filtered/ranked by the LLM.")
  metadata: typing.Dict[str, typing.Any] = Field(default_factory=dict, description="Optional execution metadata populated if `include_metadata` is True. Contains: response (raw LLM response with prompt and content) and excluded_items (list of discarded candidates).")

  @property
  def content(self): return self.metadata['response'].content

  @property
  def empty(self): return len(self.items) == 0

  @property
  def plain(self):
    if self.metadata:
      content = 'Prompt:\n-------\n' + self.metadata['response'].metadata['prompt']
      content += '\n\nResponse:\n---------\n' + self.content
    else:
      content = 'Must include metadata to see a plant result!'
    return content

  class Config:
      arbitrary_types_allowed = True
