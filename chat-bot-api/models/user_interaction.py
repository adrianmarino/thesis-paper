from pydantic import BaseModel, Field
import datetime


def str_now():
    return '{date:%Y-%m-%d %H:%M:%S}'.format( date=datetime.datetime.now() )


class UserInteraction(BaseModel):
    user_id : str = Field(..., description="The user's unique identifier (e.g., email or DB ID).", examples=["adrianmarino@gmail.com"])
    item_id : str = Field(..., description="The unique identifier of the item/movie.", examples=["tt0109830"])
    rating: float = Field(..., description="The explicit rating given by the user, usually on a scale from 1 to 5.", examples=[4.5])
    timestamp: str = Field(default_factory=str_now, description="The date and time when the interaction occurred.")
