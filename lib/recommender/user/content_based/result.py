from IPython.core.display import HTML

from rest import IMDBApiClient
from recommender import RecommenderResult, render_imdb_image


class UserProfileRecommenderResult(RecommenderResult):
    def __init__(self, recommender_name, data, k, user_id_col, item_id_col, rating_col, score_col='score'):
        self.__client = IMDBApiClient()
        self.__recommender_name = recommender_name
        self.__score_col        = score_col
        self.__user_id_col      = user_id_col
        self.__item_id_col      = item_id_col
        self.__rating_col       = rating_col

        if data.shape[0] > 0:
            self.__data = data.head(k) if k else data
        else:
            self.__data = None



    @property
    def data(self): return self.__data

    def show(self, image_width=150):
        print(f'\nItem Recommender: {self.__recommender_name}\n')

        if self.__data is None:
            print('Not Found recommendations!')
            return

        df = self.__data.copy()

        df['image'] = df['imdb_id'].apply(lambda id: render_imdb_image(self.__client, id, width=image_width))

        df = df[[self.__score_col, 'raw_score', 'popularity', self.__rating_col, 'votes', 'image', self.__user_id_col, self.__item_id_col]]
        df = df.rename(columns={
            self.__score_col: 'Score',
            'image': 'Item',
            self.__user_id_col: 'User'
        })
        df = df.reset_index(drop=True)


        display(HTML(df.to_html(escape=False)))
