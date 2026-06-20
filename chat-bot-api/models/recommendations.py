from pydantic import BaseModel, Field
from .recommendation import Recommendation
import typing


class Recommendations(BaseModel):
  items: list[Recommendation] = Field(..., description="The final list of recommended movies, retrieved via RAG/CF and filtered/ranked by the LLM.")
  metadata: typing.Dict[str, typing.Any] = Field(default_factory=dict, description="Optional metadata containing the internal LLM logs, reasoning, prompt details, and execution times. Only populated if `include_metadata` was set to True in the request.")

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
