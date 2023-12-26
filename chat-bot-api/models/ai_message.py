from .model import Model
import typing

class AIMessage(Model):
    author: str = 'AI'
    content : str
    metadata: typing.Dict[str, typing.Any] = {}


    @staticmethod
    def from_response(response):
        return AIMessage(content=response.content, metadata=response.metadata)