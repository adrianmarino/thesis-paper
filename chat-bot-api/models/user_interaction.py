from .model import Model


class UserInteraction(Model):
    user_id : str
    item_id : str
    rating: float
