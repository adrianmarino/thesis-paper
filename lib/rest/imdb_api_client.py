import requests
import logging

class IMDBApiClient:
    def __init__(self, apikey='20b2b3d5'):
        self.__apikey = apikey
    
    def __complete_id(self, id):
        return 'tt' + ( ('0' * (7 - len(id))) + id if len(id) < 7 else id )

    def get_info(self, imdb_id):
        imdb_id = str(imdb_id)
        api_url = f'http://www.omdbapi.com/?i={self.__complete_id(imdb_id)}&apikey={self.__apikey}'
        logging.debug(f'GET {api_url}')
        response = requests.get(api_url)
        return response.json()