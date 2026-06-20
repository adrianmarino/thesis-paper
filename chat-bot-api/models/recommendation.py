from pydantic import BaseModel, Field
import typing

class ResultItemMetadata(BaseModel):
  position: int = Field(..., description="The original rank position suggested by the LLM.", examples=[1])
  title: str = Field(..., description="The original title suggested by the LLM.", examples=["The Matrix"])

class DbItemMetadata(BaseModel):
  id: str = Field(..., description="The unique database identifier of the movie.", examples=["12345"])
  title: str = Field(..., description="The actual title of the movie in the database.", examples=["The Matrix"])
  release: str = Field(..., description="The release year of the movie in the database.", examples=["1999"])
  rating: float = Field(..., description="The average rating of the movie in the database.", examples=[4.8])
  query_sim: float = Field(..., description="The semantic similarity score between the query and the movie.", examples=[0.85])
  title_sim: float = Field(..., description="The cosine similarity score between the LLM's suggested title and the database title.", examples=[1.0])
  release_sim: float = Field(..., description="The similarity score between the suggested release year and the database year.", examples=[1.0])

class RecommendationMetadata(BaseModel):
  result_item: ResultItemMetadata = Field(..., description="Information about the original suggestion from the LLM.")
  db_item: DbItemMetadata = Field(..., description="Detailed technical similarity metrics from the database search (RAG).")

  def __getitem__(self, item):
    return getattr(self, item)

  def __setitem__(self, key, value):
    setattr(self, key, value)

class Recommendation(BaseModel):
  title: str = Field(..., description="The title of the recommended movie.", examples=["The Matrix"])
  poster: str | None = Field(None, description="URL of the movie's poster image.", examples=["https://example.com/poster.jpg"])
  release: str = Field(..., description="Release date or year of the movie.", examples=["1999"])
  description: str = Field(..., description="A short plot summary of the movie.", examples=["A computer hacker learns from mysterious rebels about the true nature of his reality..."])
  genres: list[str] = Field(..., description="List of genres the movie belongs to.", examples=[["Action", "Sci-Fi"]])
  votes: list[str] = Field(default_factory=list, description="List of reasons, votes, or context that justify this recommendation.")
  metadata: RecommendationMetadata | None = Field(None, description="Additional internal metadata populated if `include_metadata` is True.")
