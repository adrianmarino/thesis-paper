from .mapper import ModelMapper
from models import Item


class ItemMapper(ModelMapper):
  def to_model(self, document):
    return Item(
      id = document['item_id'],
      title = document['title'],
      genres = document['genres'],
      description = document['description'],
      release = document['release']
    )

  def to_dict(self, model):
    return {
        'item_id': model.id,
        'title': model.title,
        'genres': model.genres,
        'description': model.description,
        'release': model.release
    }