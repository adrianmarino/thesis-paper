from .mapper import ModelMapper
from models import AIMessage, UserMessage
import sys


class ChatMessageMapper(ModelMapper):
  def to_model(self, document):
    timestamp_val = document.get('timestamp')
    if document['author'] == 'AI':
      return AIMessage(
        content = document['content'],
        metadata = document['metadata'],
        **({'timestamp': timestamp_val} if timestamp_val else {})
      )
    else:
      return UserMessage(
        author = document['author'],
        content = document['content'],
        **({'timestamp': timestamp_val} if timestamp_val else {})
      )

  def to_dict(self, model):
    data = {
        'author': model.author,
        'content': model.content,
        'timestamp': model.timestamp
    }

    if isinstance(model, AIMessage):
      data['metadata'] = model.metadata

    return data
