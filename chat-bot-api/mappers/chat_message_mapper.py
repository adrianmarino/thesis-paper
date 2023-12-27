from .mapper import ModelMapper
from models import AIMessage, UserMessage


class ChatMessageMapper(ModelMapper):
  def to_model(self, document):
    if document['author'] == 'AI':
      return AIMessage(
        content = document['content'],
        metadata = document['metadata']
      )
    else:
      return UserMessage(
        author = document['author'],
        content = document['content']
      )

  def to_dict(self, model):
    data = {
        'author': model.author,
        'content': model.content
    }

    if isinstance(model, AIMessage):
      data['metadata'] = model.metadata

    return data
