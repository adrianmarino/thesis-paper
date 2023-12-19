from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from .output_parser    import MovieRecommenderDiscListOutputParser
from langchain.prompts import ChatPromptTemplate


class OllamaModelBuilder:
    def chat(self, model='movie_recommender'):
        return ChatOllama(
            model=model,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

    def default(self, model='movie_recommender'):
        return Ollama(
            model=model,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )


class OllamaChatPromptTemplateBuilder:
    def build(self, promp):
        return ChatPromptTemplate.from_messages([
            ('system', promp),
            ('human', '{query}')
        ])


class OllamaChainBuilder:
    def default(self, model, promp):
        return OllamaChatPromptTemplateBuilder().build(promp) | OllamaModelBuilder().default(model) | MovieRecommenderDiscListOutputParser()

    def chat(self, model, promp):
        return OllamaChatPromptTemplateBuilder().build(promp) | OllamaModelBuilder().chat(model)

    def chat_dict_list(self, model, promp):
        return OllamaChatPromptTemplateBuilder().build(promp) | OllamaModelBuilder().chat(model) | MovieRecommenderDiscListOutputParser()