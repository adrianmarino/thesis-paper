from .model import Model
import datetime


def str_now():
    return '{date:%Y-%m-%d %H:%M:%S}'.format( date=datetime.datetime.now() )
   



class UserInteraction(Model):
    user_id : str
    item_id : str
    rating: float
    timestamp: str = str_now()