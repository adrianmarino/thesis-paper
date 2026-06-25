from pydantic import BaseModel, Field
from datetime import datetime

class UserMessage(BaseModel):
    author: str = Field(..., description="The user's ID or email. This is crucial for retrieving the user's profile and mitigating the cold-start problem.", examples=["adrianmarino@gmail.com"])
    content: str = Field(..., description="The user's natural language query.", examples=["I want to watch a 90s sci-fi movie"])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
