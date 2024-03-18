from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama



class OllamaModelBuilder:
    @staticmethod
    def chat(model='movie_recommender', verbose=False):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) if verbose else None
        return ChatOllama(model=model, callback_manager=callback_manager)

    @staticmethod
    def default(model='movie_recommender', verbose=False):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) if verbose else None
        return Ollama(model=model, callback_manager=callback_manager)


class OllamaChatPromptTemplateFactory:
    @staticmethod
    def create(prompt):
        return ChatPromptTemplate.from_messages([
            ('system', prompt),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{request}')
        ])


class OllamaChainBuilder:
    @staticmethod
    def default(model, prompt, verbose=False):
        return OllamaChatPromptTemplateFactory.create(prompt) | \
            OllamaModelBuilder.default(model, verbose)

    @staticmethod
    def chat(model, prompt, verbose=False):
        return OllamaChatPromptTemplateFactory.create(prompt) | \
            OllamaModelBuilder.chat(model, verbose)
