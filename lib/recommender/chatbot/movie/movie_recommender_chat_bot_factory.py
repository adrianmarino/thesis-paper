from model import OllamaChainBuilder
from ..text.text_chat_bot import TextChatBot
from ..stateless.stateless_chat_bot import StatelessChatBot

from .movie_recommendations_output_parser import MovieRecommendationsOutputParser
from .movie_recommender_params_resolver import MovieRecommenderParamsResolver





TEXT_MODEL_PROMPT = """
Eres un servicio que recomienda películas a sus usuarios. Además te comunicas con tus usuarios en lenguaje español.

Quiero que recomiendes una película a un usuario basándote en información personal y
registros históricos de películas vistas.

Perfil del usuario: {user_profile}.

Los registros históricos incluyen el nombre de la película, el tipo y cuántos puntos
obtuvo de 5. Cuanto mayor es la puntuación, más le gusta la película. Te animamos a
aprender su preferencia de películas de las películas que ha visto. Aquí hay algunos
ejemplos:

{user_history}.

Aquí hay una lista de películas que probablemente le gustarán: {candidates}.

Por favor, selecciona las {limit} mejores películas de la lista que es más probable que le gusten.
La primera la película con mayor rating es la más cercana a los gustos del usuario. Por favor, selecciona las 4 películas
restantes. Solo muestra el nombre de la película y si identificador entre paréntesis.

En caso de no tener ninguna información para recomendar elige entre las películas que conozcas. Siempre responde en lenguaje español.

El formato de la respuesta debe ser siempre el mismo:

Recomendaciones:
Número. Título(Año de estreno): Descripción.

El título, año de estreno, calificación y descripción debe especificarse en lenguaje natural y no deben estar entre comillas.
"""


CHAT_BOT_PROMPT = 'Hola {user_name}, ¿Que querés que te recomiende hoy?'


class MovieRecommenderChatBotFactory:
    @staticmethod
    def text(
        model           = 'movie_recommender',
        prompt          = TEXT_MODEL_PROMPT,
        params_resolver = MovieRecommenderParamsResolver(),
        output_parser   = MovieRecommendationsOutputParser(list_size=5),
        chat_bot_prompt = CHAT_BOT_PROMPT,
        verbose         = True
    ):
        return TextChatBot(
            model,
            prompt,
            params_resolver,
            output_parser,
            chat_bot_prompt,
            verbose
        )

    @staticmethod
    def stateless(
        prompt,
        model           = 'movie_recommender',
        params_resolver = MovieRecommenderParamsResolver(),
        output_parser   = MovieRecommendationsOutputParser(list_size=5)
    ):
        return StatelessChatBot(
            model,
            prompt,
            params_resolver,
            output_parser,
        )
