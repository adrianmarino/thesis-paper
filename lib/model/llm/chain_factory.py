from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from .output_parser    import MovieRecommenderOutputParser
from .template_builder import OllamaChatPromptTemplateBuilder


class OllamaChainFactory:
    def create(self, model):
        llm = Ollama(
            model='movie_recommender',
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

        chat_prompt = OllamaChatPromptTemplateBuilder().build()

        return chat_prompt | llm | MovieRecommenderOutputParser()
