from .model import Model
from .user_interaction import UserInteraction
from .item import Item


class UserInteractionInfo(Model):
    interaction : UserInteraction
    item : Item


    @staticmethod
    def to_str(interactions_info):
        return '\n'.join([f'- {info.item.title}({"|".join(info.item.genres)}): Calificada con {info.interaction.rating} puntos.' for info in interactions_info])