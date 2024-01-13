from .ai_message import AIMessage
from .user_message import UserMessage
from .model import Model
from .chat_session import ChatSession
import typing

class ChatHistory(Model):
    email: str
    dialogue: list

    def append_dialogue(self, user_message: UserMessage, ai_message: AIMessage):
        self.dialogue.extend([user_message, ai_message])
        return self
