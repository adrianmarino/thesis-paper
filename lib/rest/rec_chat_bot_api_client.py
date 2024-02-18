import requests
import logging
import requests
import json


class RecChatBotV1ApiClient:
    def __init__(
        self,
        host    = 'nonosoft.ddns.net',
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


    def add_items(self, items):
        def to_dto(row):
            return {
                'id'          : str(row['movie_id']),
                'title'       : row['movie_title'].strip(),
                'description' : row['movie_overview'].strip(),
                'genres'      : [g.lower() for g in row['movie_genres']],
                'release'     : str(row['movie_release_year']),
                'rating'      : float(row['user_movie_rating']),
                'imdb_id'     : str(row['movie_imdb_id']),
                'poster'     : str(row['poster'])
            }
        return self._bulk_add(items, 'items/bulk', to_dto, pase_size=1000)

            
    def add_interactions(self, interactions):
        def to_dto(row):
            return {
                'user_id'   : str(row['user_id']),
                'item_id'   : str(row['item_id']),
                'rating'    : int(row['rating']),
                'timestamp' : str(row['timestamp']),
            }
        return self._bulk_add(interactions, 'interactions/bulk', to_dto, pase_size=5000)


    def _bulk_add(self, df, url, to_dto, pase_size=500):
        error_rows= []
        page        = []
        page_counter = 0
        for _, row in df.iterrows():
            if page_counter < pase_size:
                page.append(to_dto(row))
                page_counter += 1
            else:
                try:
                    headers =  {"Content-Type":"application/json"}
                    response = requests.post(f'{self.__base_url}/{url}', data=json.dumps(page), headers=headers)
                except Exception as e:
                    print(e)
                    error_rows.extend(page)
                finally:
                    page_counter = 0
                    page         = []
        return error_rows