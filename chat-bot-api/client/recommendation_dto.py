import requests
import logging
import json


class RecommendationDto:
    def __init__(self, data):
        self.__data = data

    @property
    def id(self): return self.__data['metadata']['db_item']['id']

    @property
    def title(self): return self.__data['title']

    @property
    def poster(self): return self.__data['poster']

    @property
    def release(self): return self.__data['release']

    @property
    def description(self): return self.__data['description']

    @property
    def genres(self): return self.__data['genres']

    @property
    def rating(self): return self.__data['metadata']['db_item']['rating']

    @property
    def query_sim(self): return self.__data['metadata']['db_item']['query_sim']

    def vote(self, value):
        vote_urls = self.__data['votes']

        value_url = [url for url in vote_urls if url.endswith(f'/{int(value)}')]

        if len(value_url) == 0:
            raise Exception('Invalid value!')

        value_url = value_url[0]

        logging.info(f'GET {value_url}')

        response = requests.get(value_url)

        logging.info(response.status_code)

        if  response.status_code != 204:
            raise Exception(f'Error to vote "{self.title}"({self.id}) with {value} point. Detail: {response.json()}')


    def __str__(self):
        return f"""
(
    Id         : {self.id}
    Title      : {self.title}
    Release    : {self.release}
    Genres     : {self.genres}
    Rating     : {self.rating}
    Poster     : {self.poster}
    Description: {self.description}
)
"""

    def __repr__(self): return str(self)