from pydantic import BaseModel, Field
import typing

class Recommendation(BaseModel):
  title: str = Field(..., description="The title of the recommended movie.", examples=["The Matrix"])
  poster: str | None = Field(None, description="URL of the movie's poster image.", examples=["https://example.com/poster.jpg"])
  release: str = Field(..., description="Release date or year of the movie.", examples=["1999"])
  description: str = Field(..., description="A short plot summary of the movie.", examples=["A computer hacker learns from mysterious rebels about the true nature of his reality..."])
  genres: list[str] = Field(..., description="List of genres the movie belongs to.", examples=[["Action", "Sci-Fi"]])
  votes: list[str] = Field(default_factory=list, description="List of reasons, votes, or context that justify this recommendation.")
  metadata: typing.Dict[str, typing.Any] | None = Field(None, description="Additional internal metadata populated if `include_metadata` is True. Contains: result_item (position and original title from LLM) and db_item (similarity metrics: query_sim, title_sim, release_sim).")
