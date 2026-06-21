from prompts import *
from recommender.chatbot.movie import MovieRecommendationsOutputParser, MovieRecommenderChatBotFactory
import sys
from rest.ollama import OllamaApiClient, CachedOllamaApiClient


class ChatBotPoolService:
  def __init__(
    self,
    default_prompt = 'prompt0',
    prompts = {
      'prompt0': PROMPT_LOW_INTERACTIONS,
      'prompt1': PROMPT_REQUIRED_INTERACTIONS
    },
    default_model = 'qwen2.5-coder:7b',
    models        = OllamaApiClient().models()
  ):
    self._default_model  = default_model
    self._default_prompt = default_prompt
    self._models         = models

    # Instantiate the raw client and wrap it with the CachedOllamaApiClient decorator
    self._raw_client     = OllamaApiClient()
    self._cached_client  = CachedOllamaApiClient(self._raw_client)

    output_parser = MovieRecommendationsOutputParser()
    self._chat_bots      = {
      p: {
        m: MovieRecommenderChatBotFactory.create(
          prompt        = prompts[p],
          model         = m,
          output_parser = output_parser,
          client        = self._raw_client # <-- Changed from self._cached_client to self._raw_client to bypass caching
        ) for m in models
      } for p in prompts.keys()
    }


  def available_models(self): return self._models


  def get(self, model, prompt):
    models_by_prompt = self._chat_bots.get(prompt, self._default_prompt)

    return models_by_prompt.get(model, self._default_model)
