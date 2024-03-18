from pydantic import BaseModel


class EntityEmb(BaseModel):
    id : str
    emb : list[float]
