from .model import Model
import typing

class UserMessage(Model):
    author: str
    content: str
