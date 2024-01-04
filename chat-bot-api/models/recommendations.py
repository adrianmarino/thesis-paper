from .model import Model
from .recommendation import Recommendation
import typing
import recommender as rd


class Recommendations(Model):
  items: list[Recommendation]
  response: rd.ChatBotResponse


  @property
  def content(self): return self.response.content


  @property
  def metadata(self): return self.response.metadata


  @property
  def plain(self):
    content = 'Prompt:\n-------\n'
    content += self.metadata['prompts'][0]['content']
    content += '\n\nResponse:\n---------\n\n'
    content += self.content
    return content


  class Config:
      arbitrary_types_allowed = True