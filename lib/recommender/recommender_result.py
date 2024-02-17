from abc import ABC, abstractmethod


def to_image_html(path, width=360, alt='Not Found Image'): return F'<img src="{path}" width="{width}" alt={alt} >'


def render_imdb_image(client, id, width=360):
    try:
        info = client.get_info(id)
        return to_image_html(info['Poster'], width)
    except:
        return 'Not Found Image'


class RecommenderResult(ABC):
    @abstractmethod
    def show(self):
        pass

    @property
    def data(self):
        pass