from .ai_message import AIMessage
from .user_message import UserMessage
from pydantic import BaseModel


class ChatHistory(BaseModel):
    email: str
    dialogue: list

    def append_dialogue(self, user_message: UserMessage, ai_message: AIMessage):
        self.dialogue.extend([user_message, ai_message])
        return self
