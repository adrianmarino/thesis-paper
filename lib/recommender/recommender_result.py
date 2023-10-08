from abc import ABC, abstractmethod


def to_image_html(path, width=360): return F'<img src="{path}" width="{width}" >'


def render_image(client, id, width=360):
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