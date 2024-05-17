from pydantic import BaseModel
from .recommendation import Recommendation
import typing


class Recommendations(BaseModel):
  items: list[Recommendation]
  metadata: typing.Dict[str, typing.Any] = {}

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