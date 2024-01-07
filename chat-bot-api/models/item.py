from .model import Model


class Item(Model):
    id : str
    imdb_id: str = None
    title : str
    description: str
    release: str
    genres: list[str]
    rating: float = None
    poster: str
    embedding : list[float] = None

    def with_embedding(self, embedding):
        return Item(
            id=self.id,
            imdb_id=self.imdb_id,
            title=self.title,
            description=self.description,
            genres=self.genres,
            release=self.release,
            rating=self.rating,
            poster=self.poster,
            embedding=embedding
        )