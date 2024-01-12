import requests
import logging

class TMDBApiClient:
    def __init__(self, api_key='15d2ea6d0dc1d476efbca3eba2b9bbfb'):
        self.__api_key = api_key

    def find_movies_by(self, query):
        try:
            movies = self._request_by_query(query)
            return movies['results']
        except Exception as e:
            return []

    def find_poster_by(self, query):
        movies = self.find_movies_by(query)
        return f"http://image.tmdb.org/t/p/w500{movies[0]['poster_path']}" if len(movies) > 0 else None


    def _request_by_query(self, query):
        api_url = f'https://api.themoviedb.org/3/search/movie?api_key={self.__api_key}&query={query}'
        logging.debug(f'GET {api_url}')
        response = requests.get(api_url)
        return response.json()
