from .mapper import ModelMapper
from .chat_message_mapper import ChatMessageMapper
from models import ChatSession

class ChatSessionMapper(ModelMapper):
  
  msg_mapper = ChatMessageMapper()

  def to_model(self, document):
    return ChatSession(
      dialogue=[self.msg_mapper.to_model(m) for m in document['dialogue']]
    )

  def to_dict(self, model):
    return {
      'dialogue': [self.msg_mapper.to_dict(m) for m in model.dialogue] 
    }
