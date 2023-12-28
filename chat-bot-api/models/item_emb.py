from .model import Model


class ItemEmb(Model):
    id : str
    emb : list[float]
