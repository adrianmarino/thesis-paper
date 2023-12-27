from .mapper import ModelMapper
from .chat_session_mapper import ChatSessionMapper
from models import ChatHistory



class ChatHistoryMapper(ModelMapper):
  session_mapper = ChatSessionMapper()

  def to_model(self, document):
    return ChatHistory(
      email = document['email'],
      sessions = [self.session_mapper.to_model(s) for s in document['sessions']]
    )

  def to_dict(self, model):
    return {
      'email': model.email,
      'sessions': [self.session_mapper.to_dict(s) for s in model.sessions]
    }
