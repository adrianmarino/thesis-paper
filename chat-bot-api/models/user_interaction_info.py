from .model import Model
from .user_interaction import UserInteraction
from .item import Item


class UserInteractionInfo(Model):
    interaction : UserInteraction
    item : Item

    @property
    def id(self): return f"{self.user_id}-{self.item_id}"


    @property
    def item_id(self): return self.item.id


    @property
    def user_id(self): return self.interaction.user_id


    @property
    def rating(self): return self.interaction.rating


    @property
    def genres(self): return self.item.genres


    @property
    def title(self): return self.item.title


    def __str__(self):
        return f'- {self.title}({"|".join(self.genres)}): Calificada con {self.rating} puntos.'


    def __hash__(self):
        return hash(self.id)


    def __eq__(another):
        return self.id == self.id if another else False