from .model import Model


class Item(Model):
    id : str
    title : str
    description: str
    release: str