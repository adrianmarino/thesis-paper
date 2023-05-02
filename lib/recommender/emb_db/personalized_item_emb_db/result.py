from IPython.core.display import HTML

from rest import IMDBApiClient
from recommender import RecommenderResult


def to_image_html(path, width=360): return F'<img src="{path}" width="{width}" >'


def render_image(client, id, width=360):
    try:
        info = client.get_info(id)
        return to_image_html(info['Poster'], width)
    except:
        return 'Not Found Image'


class PersonalizedItemEmbDBRecommenderResult(RecommenderResult):
    def __init__(self, data):
        self.__client = IMDBApiClient()
        self.__data = data

    @property
    def data(self):
        return self.__data

    def show(self, image_width=300):
        df = self.__data.copy()

        df['image'] = df['imdb_id'].apply(lambda id: render_image(self.__client, id, width=image_width))
        df['sim_image'] = df['sim_imdb_id'].apply(lambda id: render_image(self.__client, id, width=image_width))

        df['.'] = 'We Recommend ==>'

        df['..'] = '==> Because You Saw ==>'

        df = df[['sim', 'sim_rating', '.', 'sim_image', '..', 'image', 'rating']]

        df = df.round({'sim': 2, 'sim_rating': 1, 'rating': 1})

        df = df.rename(columns={
            'sim_rating': 'Rating',
            'sim': 'Similarity',
            'sim_image': 'Recommended Movies',
            'image': 'Already seen movies',
            'rating': 'Rating'
        })

        display(HTML(df.to_html(escape=False)))
