from .model import Model


class Item(Model):
    id : str
    title : str
    description: str
    release: str
    genres: list[str]
    embedding : list[float] = None

    def with_embedding(self, embedding):
        return Item(
            id=self.id,
            title=self.title,
            description=self.description,
            genres=self.genres,
            release=self.release,
            embedding=embedding
        )