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
    default_model = 'qwen2.5-coder:7b'
  ):
    self._default_model  = default_model
    self._default_prompt = default_prompt
    self._prompts        = prompts

    # Instantiate the raw client and wrap it with the CachedOllamaApiClient decorator
    self._raw_client     = OllamaApiClient()
    self._cached_client  = CachedOllamaApiClient(self._raw_client)
    self._output_parser  = MovieRecommendationsOutputParser()

    # Empty dictionary of bots for lazy initialization
    self._chat_bots      = {
      p: {} for p in prompts.keys()
    }


  def available_models(self):
    try:
      return self._raw_client.models()
    except Exception as e:
      print(f"Error fetching Ollama models: {e}")
      return [self._default_model]


  def get(self, model, prompt):
    # Fallback to default prompt if requested one is not found
    if prompt not in self._prompts:
      prompt = self._default_prompt

    # Lazy-load the chat bot if it hasn't been instantiated yet for this prompt-model pair
    if model not in self._chat_bots[prompt]:
      try:
        # Create and cache the bot
        self._chat_bots[prompt][model] = MovieRecommenderChatBotFactory.create(
          prompt        = self._prompts[prompt],
          model         = model,
          output_parser = self._output_parser,
          client        = self._cached_client
        )
      except Exception as e:
        print(f"Failed to create chat bot for model {model}: {e}")
        # Fallback to the default model
        if self._default_model not in self._chat_bots[prompt]:
          self._chat_bots[prompt][self._default_model] = MovieRecommenderChatBotFactory.create(
            prompt        = self._prompts[prompt],
            model         = self._default_model,
            output_parser = self._output_parser,
            client        = self._cached_client
          )
        return self._chat_bots[prompt][self._default_model]

    return self._chat_bots[prompt][model]
