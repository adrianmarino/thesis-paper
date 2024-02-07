from .mapper import ModelMapper
from models import UserInteraction


class InteractionMapper(ModelMapper):
  def to_model(self, document):
    return UserInteraction(
      user_id   = str(document['user_id']),
      item_id   = document['item_id'],
      rating    = document['rating'],
      timestamp = document['timestamp']
    )

  def to_dict(self, model):
    return {
        'user_id': model.user_id,
        'item_id': model.item_id,
        'rating': model.rating,
        'timestamp': model.timestamp
    }