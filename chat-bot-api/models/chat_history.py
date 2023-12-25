from .ai_message import AIMessage
from .user_message import UserMessage
from .model import Model
from .chat_session import ChatSession
import typing


class ChatHistory(Model):
    email: str
    sessions: list[ChatSession]

    def as_content_list(self):
        result = []
        for s in self.sessions:
            for i in range(0, len(s.dialogue), 2):
                user, ai = s.dialogue[i:i+2]
                result.append((user.content, ai.content))
        return result

    def append_dialogue(self, user_message: UserMessage, ai_message: AIMessage):
        self.sessions[-1].dialogue.extend([user_message, ai_message])
        return self