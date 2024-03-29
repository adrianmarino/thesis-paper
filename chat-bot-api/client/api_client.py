import requests
import logging
import requests
import json
from .recommendations_dto import RecommendationsDto
from .user_profile_dto    import UserProfileDto
from bunch import Bunch
from .exceptions          import *


class RecChatBotV1ApiClient:
    def __init__(
        self,
        host      = 'nonosoft.ddns.net',
        port      = 8080,
        timeout   = 30,
        verbose   = True
    ):
        self.__base_url = f'http://{host}:{port}/api/v1'
        self.__timeout = timeout
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__verbose = verbose


    def _log(self, value):
        if self.__verbose:
            self.__logger.info(value)

    @property
    def verbose_off(self):
        self.__verbose = False

    @property
    def verbose_on(self):
        self.__verbose = True

    @property
    def health(self):
        api_url = f'{self.__base_url}/health'

        self._log(f'GET {api_url}')

        response = requests.get(api_url, timeout=self.__timeout)

        if  response.status_code != 200:
            raise Exception(f'Error when get health. Detail: {response.json()}')

        return response.json()


    def remove_interactions_by_user_id(self, user_id):
        api_url = f'{self.__base_url}/interactions/users/{user_id}'
        self._log(f'DELETE {api_url}')

        response = requests.delete(api_url, timeout=self.__timeout)

        if  response.status_code != 202:
            raise Exception(f'Error when remove user interactions. Detail: {response.json()}')



    def add_profile(self, profile: UserProfileDto)->UserProfileDto:
        api_url = f'{self.__base_url}/profiles'
        self._log(f'POST {api_url}')
        response = requests.post(
            api_url,
            data    = profile.to_json(),
            headers = { 'Content-Type': 'application/json' },
            timeout = self.__timeout
        )

        if  response.status_code != 204:
            raise Exception(f'Error when add user profile. Detail: {response.json()}')


    def delete_profile(self, email: str):
        api_url = f'{self.__base_url}/profiles/{email}'
        self._log(f'DELETE {api_url}')
        response = requests.delete(api_url, timeout=self.__timeout)

        if  response.status_code != 202:
            raise Exception(f'Error when delete user profile. Detail: {response.json()}')


    def profiles(self)-> list[UserProfileDto]:
        api_url = f'{self.__base_url}/profiles'
        self._log(f'GET {api_url}')
        response = requests.get(api_url, timeout=self.__timeout)

        if  response.status_code != 200:
            raise Exception(f'Error when query user profile. Detail: {response.json()}')

        return [UserProfileDto.from_json(data) for data in response.json()]


    def recommendations(self, query)-> list[RecommendationsDto]:
        api_url = f'{self.__base_url}/recommendations'
        self._log(f'POST {api_url}')
        response = requests.post(
            api_url,
            data    = json.dumps(query),
            headers = { 'Content-Type': 'application/json' }
        )

        if  response.status_code == 404:
            raise NotFoundException(response.content)

        if  response.status_code != 200:
            raise Exception(response)


        return RecommendationsDto(response.json(), self.__verbose)


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
        self._log(f'GET {api_url}')

        response = requests.get(api_url)

        if  response.status_code != 200:
            raise Exception(f'Error when query items by {criterion} criterion. Detail: {response.json()}')

        return response.json()


    def interactions_by_user(
        self,
        email    : str  = '',
    ):
        api_url = f'{self.__base_url}/interactions/users/{email}'
        self._log(f'GET {api_url}')
        response = requests.get(api_url, timeout=self.__timeout)

        if  response.status_code == 404:
            return []

        if  response.status_code != 200:
            raise Exception(f'Error when query interactions by "{email}" user id. Detail: {response.json()}')

        return [Bunch(item) for item in response.json()]



    def interactions(self):
        api_url = f'{self.__base_url}/interactions'
        self._log(f'GET {api_url}')
        response = requests.get(api_url, timeout=self.__timeout)

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

        self._log(f'Page Size: {page_size}')

        for page in range(n_pages):
            page_df = df.iloc[page_size * page : page_size * (page + 1)]
            dtos = [to_dto(row) for idx, row in  page_df.iterrows()]

            try:
                response = requests.post(
                    f'{self.__base_url}/{url}',
                    data    = json.dumps(dtos),
                    headers = {
                        'Content-Type': 'application/json'
                    },
                    timeout=self.__timeout
                )

                if  response.status_code == 201:
                    self._log(f'Page: {1+page}/{n_pages}, Size: {len(dtos)}')
                else:
                    error_rows.extend(page_df)

            except Exception as error:
                print(error)
                error_rows.extend(page_df)

        return error_rows
