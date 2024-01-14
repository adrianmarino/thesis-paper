from prompts import *
import recommender as rd


class ChatBotPoolService:
  def __init__(self):
    output_parser = rd.MovieRecommendationsOutputParser(list_size=15)

    self.with_candidates = {
      'llama2-7b-chat': rd.MovieRecommenderChatBotFactory.stateless(
        model         = 'llama2_7b_chat_model',
        prompt        = PROMPT_WITH_CANDIDATES,
        output_parser = output_parser
      ),
      'llama2-13b-chat': rd.MovieRecommenderChatBotFactory.stateless(
        model         = 'llama2_13b_chat_model',
        prompt        = PROMPT_WITH_CANDIDATES,
        output_parser = output_parser
      ),
      'mixtral': rd.MovieRecommenderChatBotFactory.stateless(
        model         = 'mixtral_model',
        prompt        = PROMPT_WITH_CANDIDATES,
        output_parser = output_parser
      ),
      'mistral': rd.MovieRecommenderChatBotFactory.stateless(
        model         = 'mistral_model',
        prompt        = PROMPT_WITH_CANDIDATES,
        output_parser = output_parser
      )
    }

    self.without_candidates = {
      'llama2-7b-chat': rd.MovieRecommenderChatBotFactory.stateless(
        model         = 'llama2_7b_chat_model',
        prompt        = PROMPT_WITHOUT_CANDIDATES,
        output_parser = output_parser
      ),
      'llama2-13b-chat': rd.MovieRecommenderChatBotFactory.stateless(
        model         = 'llama2_13b_chat_model',
        prompt        = PROMPT_WITHOUT_CANDIDATES,
        output_parser = output_parser
      ),
      'mixtral': rd.MovieRecommenderChatBotFactory.stateless(
        model         = 'mixtral_model',
        prompt        = PROMPT_WITHOUT_CANDIDATES,
        output_parser = output_parser
      ),
      'mistral': rd.MovieRecommenderChatBotFactory.stateless(
        model         = 'mistral_model',
        prompt        = PROMPT_WITHOUT_CANDIDATES,
        output_parser = output_parser
      )
    }


  def available_models(self):
    return [
      'mistral',
      'llama2-7b-chat',
      'mixtral',
      'llama2-13b-chat',
    ]

  def get(self, model, with_candidates):
    if model not in self.with_candidates or model not in self.without_candidates:
      raise Exception(f'Missing {model} model')

    if with_candidates:
      return self.with_candidates[model]
    else:
      return self.without_candidates[model]
