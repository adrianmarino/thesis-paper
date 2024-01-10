from prompts import *
import recommender as rd


class ChatBotPoolService:
  def __init__(self):
    output_parser = rd.MovieRecommendationsOutputParser(list_size=15)

    self.with_candidates = {
      'ollama2': rd.MovieRecommenderChatBotFactory.stateless(
        prompt        = PROMPT_WITH_CANDIDATES,
        output_parser = output_parser
      )
    }

    self.without_candidates = {
      'ollama2': rd.MovieRecommenderChatBotFactory.stateless(
        prompt        = PROMPT_WITHOUT_CANDIDATES,
        output_parser = output_parser
      )
    }


  def get(self, model, with_candidates):
    if model not in self.with_candidates or model not in self.without_candidates:
      raise Exception(f'Missing {model} model')

    if with_candidates:
      return self.with_candidates[model]
    else:
      return self.without_candidates[model]
