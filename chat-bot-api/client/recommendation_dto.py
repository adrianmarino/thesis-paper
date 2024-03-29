import requests
import logging
import json


class RecommendationDto:
    def __init__(self, data, verbose):
        self.__data    = data
        self.__verbose = verbose
        self.__logger = logging.getLogger(self.__class__.__name__)


    def _log(self, value):
        if self.__verbose:
            self.__logger.info(value)

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

        self._log(f'GET {value_url}')

        response = requests.get(value_url)

        self._log(response.status_code)

        if  response.status_code != 204:
            raise Exception(f'Error to vote "{self.title}"({self.id}) with {value} point. Detail: {response.json()}')

    def to_json(self):
        return json.dumps(
            self.__data,
            default=lambda o: o.__dict__,
            sort_keys=True, indent=4
        )

    def __str__(self):
        return self.to_json()

    def __repr__(self): return str(self)