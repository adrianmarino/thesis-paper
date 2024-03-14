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


    def remove_interactions_by_user_id(self, user_id):
        api_url = f'{self.__base_url}/interactions/users/{user_id}'
        logging.info(f'DELETE {api_url}')

        response = requests.delete(api_url)

        if  response.status_code != 202:
            raise Exception(f'Error when remove user interactions!. Detail: {response.json()}')


    def recommendations(self, query):
        api_url = f'{self.__base_url}/recommendations'
        logging.info(f'POST {api_url}')
        response = requests.post(
            api_url,
            data    = json.dumps(query),
            headers = { 'Content-Type': 'application/json' }
        )

        if  response.status_code != 200:
            raise Exception(f'Error when query recommendations by query: {query.json()}. Detail: {response.json()}')

        return response.json()


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
        criterion = f'email={email}&seen={seen}&content={content}&all={all}&limit={limit}&hide_emb={hide_emb}&release={release}&genres={genres}'
        api_url = f'{self.__base_url}/items?{criterion}'
        logging.debug(f'GET {api_url}')

        response = requests.get(api_url)

        if  response.status_code != 200:
            raise Exception(f'Error when query items by {criterion} criterion. Detail: {response.json()}')

        return response.json()


    def interactions_by_user(
        self,
        email    : str  = '',
    ):
        api_url = f'{self.__base_url}/interactions/users/{email}'
        logging.debug(f'GET {api_url}')
        response = requests.get(api_url)

        if  response.status_code != 404:
            return []

        if  response.status_code != 200:
            raise Exception(f'Error when query interactions by "{email}" user id. Detail: {response.json()}')

        return response.json()



    def interactions(self):
        api_url = f'{self.__base_url}/interactions'
        logging.debug(f'GET {api_url}')
        response = requests.get(api_url)

        if  response.status_code != 200:
            raise Exception(f'Error when query interactions!. Detail: {response.json()}')

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
        return self._bulk_add(items, 'items/bulk', to_dto, page_size=1000)


    def add_interactions(self, interactions):
        def to_dto(row):
            return {
                'user_id'   : str(row['user_id']),
                'item_id'   : str(row['item_id']),
                'rating'    : int(row['rating']),
                'timestamp' : str(row['timestamp']),
            }
        return self._bulk_add(interactions, 'interactions/bulk', to_dto, page_size=5000)


    def _bulk_add(
        self,
        df,
        url,
        to_dto,
        page_size = 500
    ):
        error_rows = []
        n_pages    = len(df) // page_size + 1

        logging.info(f'Page Size: {page_size}')

        for page in range(n_pages):
            page_df = df.iloc[page_size * page : page_size * (page + 1)]
            dtos = [to_dto(row) for idx, row in  page_df.iterrows()]

            try:
                response = requests.post(
                    f'{self.__base_url}/{url}',
                    data    = json.dumps(dtos),
                    headers = {
                        'Content-Type': 'application/json'
                    }
                )

                if  response.status_code == 201:
                    logging.info(f'Page: {1+page}/{n_pages}, Size: {len(dtos)}')
                else:
                    error_rows.extend(page_df)

            except Exception as error:
                print(error)
                error_rows.extend(page_df)

        return error_rows
