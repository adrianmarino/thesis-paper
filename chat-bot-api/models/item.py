from pydantic import BaseModel, Field


class Item(BaseModel):
    id : str = Field(..., description="Internal unique identifier for the item.", examples=["12345"])
    imdb_id: str = Field(None, description="External IMDB identifier for the movie.", examples=["tt0109830"])
    title : str = Field(..., description="The title of the movie.", examples=["Forrest Gump"])
    description: str = Field(..., description="The plot summary or overview of the movie. Very important for semantic RAG search.", examples=["The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate, and other history unfold through the perspective of an Alabama man with an IQ of 75."])
    release: str = Field(..., description="Release date or year.", examples=["1994"])
    genres: list[str] = Field(..., description="List of genres associated with the movie.", examples=[["Drama", "Romance"]])
    rating: float = Field(None, description="Average rating of the item across all users.", examples=[4.8])
    poster: str = Field(..., description="URL of the movie's poster image.", examples=["https://example.com/poster.jpg"])
    embedding : list[float] = Field(None, description="The pre-calculated sentence embedding vector representing the movie's content.")

    def with_embedding(self, embedding):
        return Item(
            id          =   str(self.id),
            imdb_id     =   str(self.imdb_id),
            title       =   self.title,
            description =   self.description,
            genres      =   self.genres,
            release     =   self.release,
            rating      =   self.rating,
            poster      =   self.poster,
            embedding   =   embedding
        )
