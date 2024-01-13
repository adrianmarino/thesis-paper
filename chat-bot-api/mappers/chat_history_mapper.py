from .mapper import ModelMapper
from .chat_message_mapper import ChatMessageMapper
from models import ChatHistory
import sys


class ChatHistoryMapper(ModelMapper):
  msg_mapper = ChatMessageMapper()

  def to_model(self, document):
    return ChatHistory(
      email = document['email'],
      dialogue=[self.msg_mapper.to_model(m) for m in document['dialogue']]
    )

  def to_dict(self, model):
    return {
      'email': model.email,
      'dialogue': [self.msg_mapper.to_dict(m) for m in model.dialogue]
    }
