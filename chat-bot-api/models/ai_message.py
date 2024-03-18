import typing
from .user_message import UserMessage
import langchain.schema
from pydantic import BaseModel


class AIMessage(BaseModel):
    author: str = 'AI'
    content : str
    metadata: typing.Dict[str, typing.Any] = {}

    @staticmethod
    def from_response(response, author):
        if 'params' in response.metadata:
            if 'chat_history' in response.metadata['params']:
                messages = response.metadata['params']['chat_history']
                mapper = LangChainMessageMapper()
                response.metadata['params']['chat_history'] = mapper.to_models(messages, author) 

        return AIMessage(content=response.content, metadata=response.metadata)


class LangChainMessageMapper:

    def to_lang_chain_messages(self, messages):
        return [self.to_lang_chain(msg) for msg in messages]

    def to_models(self, messages, author):
        return [self.to_data(msg, author) for msg in messages]


    def to_lang_chain(self, msg):
        if  msg.author == 'AI':
            return langchain.schema.AIMessage(content=msg.content)
        else:
            return langchain.schema.HumanMessage(content=msg.content)

    def to_data(self, msg, author):
        if  type(msg) == langchain.schema.AIMessage:
            return {'author': 'AI', 'content': msg.content}
        else:
            return {'author': author, 'content': msg.content}