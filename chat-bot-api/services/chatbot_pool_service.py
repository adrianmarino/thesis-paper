from promps import *
import recommender as rd


class ChatBotPoolService:
  def __init__(self):
    output_parser = rd.MovieRecommendationsOutputParser(list_size=10)

    self.with_candidates = rd.MovieRecommenderChatBotFactory.stateless(
      prompt        = PROMPT_WITH_CANDIDATES,
      output_parser = output_parser
    )

    self.without_candidates = rd.MovieRecommenderChatBotFactory.stateless(
      prompt = PROMPT_WITHOUT_CANDIDATES,
      output_parser = output_parser
    )
