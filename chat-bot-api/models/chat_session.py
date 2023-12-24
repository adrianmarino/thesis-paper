from .model import Model
from .ai_message import AIMessage
from .user_message import UserMessage
import typing

class ChatSession(Model):
    dialogue: list
