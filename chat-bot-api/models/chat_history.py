from .ai_message import AIMessage
from .user_message import UserMessage
from pydantic import BaseModel, Field


class ChatHistory(BaseModel):
    email: str = Field(..., description="The email identifying the user who owns this chat history.", examples=["adrianmarino@gmail.com"])
    dialogue: list = Field(..., description="The chronological list of messages exchanged between the user and the AI.")

    def append_dialogue(self, user_message: UserMessage, ai_message: AIMessage):
        self.dialogue.extend([user_message, ai_message])
        return self
