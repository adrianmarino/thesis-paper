from .model import Model


class EntityEmb(Model):
    id : str
    emb : list[float]
