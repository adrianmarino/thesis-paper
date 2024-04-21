from ..chat_bot_client import ChatBotClient

from .movie_recommendations_output_parser import MovieRecommendationsOutputParser
from .movie_recommender_params_resolver import MovieRecommenderParamsResolver

class MovieRecommenderChatBotFactory:
    @staticmethod
    def create(
        prompt,
        model,
        params_resolver = MovieRecommenderParamsResolver(),
        output_parser   = MovieRecommendationsOutputParser(size=5)
    ):
        return ChatBotClient(
            model,
            prompt,
            params_resolver,
            output_parser,
        )
