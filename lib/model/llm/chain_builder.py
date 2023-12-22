from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate


class OllamaModelBuilder:
    @staticmethod
    def chat(model='movie_recommender'):
        return ChatOllama(
            model=model,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

    @staticmethod
    def default(model='movie_recommender'):
        return Ollama(
            model=model,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )


class OllamaChatPromptTemplateFactory:
    @staticmethod
    def create(prompt):
        return ChatPromptTemplate.from_messages([
            ('system', prompt),
            ('human', '{input}')
        ])


class OllamaChainBuilder:
    @staticmethod
    def default(model, prompt):
        return OllamaChatPromptTemplateFactory.create(prompt) | \
            OllamaModelBuilder.default(model)

    @staticmethod
    def chat(model, prompt):
        return OllamaChatPromptTemplateFactory.create(prompt) | \
            OllamaModelBuilder.chat(model)
