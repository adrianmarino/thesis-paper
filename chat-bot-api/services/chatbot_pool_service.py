from promps import *
import recommender as rd


class ChatBotPoolService:
  def __init__(self):
    self.with_candidates = rd.MovieRecommenderChatBotFactory.stateless(PROMPT_WITH_CANDIDATES)
    self.without_candidates = rd.MovieRecommenderChatBotFactory.stateless(PROMPT_WITHOUT_CANDIDATES)
