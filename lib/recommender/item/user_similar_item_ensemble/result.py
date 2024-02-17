from IPython.core.display import HTML
from rest import IMDBApiClient
from recommender import RecommenderResult, render_imdb_image


class UserSimilarItemEnsembleRecommenderResult(RecommenderResult):
    def __init__(self, recommender_name, data, k):
        self.__client = IMDBApiClient()
        self.__recommender_name = recommender_name
        self.__data = data.sort_values(by=['sim_rating'], ascending=False).head(k) if data.shape[0] > 0 else None

    @property
    def data(self): return self.__data

    def show(self, image_width=300):
        print(f'\nRecommender: {self.__recommender_name}\n')

        if self.__data is None:
            print('Not Found recommendations!')
            return

        df = self.__data.copy()

        df['sim_image'] = df['sim_imdb_id'].apply(lambda id: render_imdb_image(self.__client, id, width=image_width))

        df = df[['sim_rating', 'sim_image', 'sim']]

        df = df.round({'sim': 2, 'sim_rating': 1})

        df = df.rename(columns={
            'sim_rating': 'Rating',
            'sim': 'Similarity',
            'sim_image': 'Recommended Movies'
        })

        display(HTML(df.to_html(escape=False)))
