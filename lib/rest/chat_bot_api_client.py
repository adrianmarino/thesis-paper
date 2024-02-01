import requests
import logging


class ChatBotV1ApiClient:
    def __init__(
        self,
        host    = 'localhost',
        port    = 8080
    ):
        self.__base_url = f'http://{host}:{port}/api/v1'

    def items(
        self,
        email    : str  = '',
        seen     : bool = True,
        content  : str  = '',
        all      : bool = False,
        limit    : int  = 5,
        hide_emb : bool = True,
        release  : int  = 1950,
        genres   : str  = ''
    ):
        api_url = f'{self.__base_url}/items?email={email}&seen={seen}&content={content}&all={all}&limit={limit}&hide_emb={hide_emb}&release={release}&genres={genres}'
        logging.debug(f'GET {api_url}')
        response = requests.get(api_url)
        return response.json()


    def interactions_by_user(
        self,
        email    : str  = '',
    ):
        api_url = f'{self.__base_url}/interactions/users/{email}'
        logging.debug(f'GET {api_url}')
        response = requests.get(api_url)
        return response.json()

    def interactions(self):
        api_url = f'{self.__base_url}/interactions'
        logging.debug(f'GET {api_url}')
        response = requests.get(api_url)
        return response.json()