import requests
import logging
import data as dt 
import util as ut


class TMDBApiClient:
    def __init__(self, api_key='15d2ea6d0dc1d476efbca3eba2b9bbfb'):
        self.__api_key = api_key
        self.__genres = self.genres


    def find_many_movies_by(self, title_by_id, fields=[]):
        movies   = {}
        with dt.progress_bar(len(title_by_id), title='Fetching many movies from TMDB') as pb:
            for id, title in title_by_id.items():
                movies[id] = self.find_movies_by(title, fields)
                pb.update()

        return movies


    def resolve_data(self, id, title):
        return (id, self.find_movies_by(title))


    def parallel_find_many_movies_by(self, title_by_id, fields=[]):        

        movie_by_id = ut.ParallelExecutor()(
            self.resolve_data,
            params          = [[id, title] for id, title in title_by_id.items()],
            fallback_result = None
        )

        return {movie[0]: movie[1] for movie in movie_by_id if movie}


    def find_movies_by(self, query, fields=[]):
        try:
            movies = self._request_by_query(query)['results']

            select_movies =[]
            for movie in movies:
                movie = self.__transform_fields(movie)
                if len(fields) > 0:
                    select_movies.append({field: movie[field] for field in fields})

            return select_movies if len(fields) > 0 else movies

        except Exception as e:
            return []


    def find_first_poster_by(self, query):
        movies = self.find_movies_by(query)
        return movies[0]['poster_path'] if len(movies) > 0 else None


    def _request_by_query(self, query):
        api_url = f'https://api.themoviedb.org/3/search/movie?api_key={self.__api_key}&query={query}'
        logging.debug(f'GET {api_url}')
        response = requests.get(api_url)
        return response.json()


    @property
    def genres(self):
        api_url = f'https://api.themoviedb.org/3/genre/movie/list?api_key={self.__api_key}'
        logging.debug(f'GET {api_url}')
        response = requests.get(api_url)
        return to_genres_model(response.json())


    @property
    def fields(self):
        movies = self.find_movies_by('Toy Story')
        return list(movies[0].keys()) if len(movies) >0 else []


    def __transform_fields(self, movie):
        movie['genres']       = [self.__genres[genre_id].lower() for genre_id in movie['genre_ids'] if genre_id in self.__genres]
        del movie['genre_ids']

        movie['poster_url']   = f'http://image.tmdb.org/t/p/w500{movie["poster_path"]}'
        del movie['poster_path']

        movie['backdrop_url'] = f'http://image.tmdb.org/t/p/w500{movie["backdrop_path"]}'
        del movie["backdrop_path"]

        return movie


def to_genres_model(dto):
    if 'genres' in dto:
        return {genre['id']:genre['name'] for genre in dto['genres']}
    else:
        return {}