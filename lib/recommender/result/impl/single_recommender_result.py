from ..recommender_result import RecommenderResult
from rest import IMDBApiClient
from IPython.core.display import HTML


def to_image_html(path,  width=160): return F'<img src="{path}" width="{width}" >'


def render_image(client, id):
    try:
        info = client.get_info(id)
        return to_image_html(info['Poster'])
    except:
        return 'Not Found Image'

class SingleRecommenderResult(RecommenderResult):
    def __init__(self, name, item, recommendations):
        self.__client = IMDBApiClient()
        self.name = name
        self.item = item
        self.recommendations = recommendations

    def show(self):
        print(f'\nRecommender: {self.name}')
        print(f'Item')
        self.__show_table(self.item, drop=['imdb_id', 'id'])

        print(f'Recommendations')
        self.__show_table(self.recommendations)

    def __show_table(self, df, drop=['imdb_id']):
        df = df.copy()
        df['image'] = df['imdb_id'].apply(lambda id: render_image(self.__client, id))
        display(HTML(df.drop(columns=drop).to_html(escape=False)))