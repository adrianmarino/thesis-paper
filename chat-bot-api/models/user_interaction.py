from pydantic import BaseModel
import datetime


def str_now():
    return '{date:%Y-%m-%d %H:%M:%S}'.format( date=datetime.datetime.now() )



class UserInteraction(BaseModel):
    user_id : str
    item_id : str
    rating: float
    timestamp: str = str_now()