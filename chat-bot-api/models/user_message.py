from pydantic import BaseModel

class UserMessage(BaseModel):
    author: str
    content: str
