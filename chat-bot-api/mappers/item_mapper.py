from .mapper import ModelMapper
from models import Item


class ItemMapper(ModelMapper):
  def to_model(self, document):
    return Item(
      id          = document['item_id'],
      title       = document['title']       if 'title' in  document else None,
      genres      = document['genres']      if 'genres' in document else [],
      description = document['description'] if 'description' in  document else None,
      rating      = document['rating']      if 'rating' in  document else None,
      release     = document['release']     if 'release' in  document else None,
      poster      = document['poster']      if 'poster' in  document else None
    )

  def to_dict(self, model):
    return {
        'item_id'    : model.id,
        'title'      : model.title,
        'genres'     : model.genres,
        'description': model.description,
        'rating'     : model.rating,
        'release'    : model.release,
        'poster'     : model.poster
    }