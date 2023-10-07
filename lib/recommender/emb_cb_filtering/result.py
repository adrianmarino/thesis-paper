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


class EmbCBFilteringRecommenderResult(RecommenderResult):
    def __init__(self, recommender_name, data,imdb_id_col, rating_col, metadata):
        self.__client           = IMDBApiClient()
        self.__recommender_name = recommender_name
        self.__imdb_id_col      = imdb_id_col
        self.__rating_col       = rating_col
        self.__data             = data
        self.__metadata         = metadata

    @property
    def data(self):
        return self.__data

    def show(self, image_width=300):
        print(f'\nRecommender: {self.__recommender_name}\n')

        if self.__data is None:
            print('Not Found recommendations!')
            return

        df = self.__data.copy()

        df['image'] = df[self.__imdb_id_col].apply(lambda id: render_image(self.__client, id, width=image_width))

        df = df[[self.__rating_col, 'image']]

        df = df.rename(columns={
            self.__rating_col: 'Rating',
            'image': 'Movies'
        })

        for mc in self.__metadata:
            df[mc] = self.__data[mc]

        display(HTML(df.to_html(escape=False)))
