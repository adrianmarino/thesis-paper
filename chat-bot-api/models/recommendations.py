from .model import Model
from .recommendation import Recommendation
import typing


class Recommendations(Model):
  items: list[Recommendation]
  metadata: typing.Dict[str, typing.Any] = {}

  @property
  def content(self): return self.metadata['response'].content

  @property
  def empty(self): return len(self.items) == 0

  @property
  def plain(self):
    content = 'Prompt:\n-------\n'
    content += self.metadata['response'].metadata['prompts'][0]['content']
    content += '\n\nResponse:\n---------\n'
    content += self.content
    return content


  class Config:
      arbitrary_types_allowed = True