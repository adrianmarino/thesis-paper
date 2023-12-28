from .mapper      import ModelMapper
from models       import ItemEmb
from bunch import Bunch
from sentence_transformers import SentenceTransformer

class ItemEmbMapper(ModelMapper):
  def __init__(self, item_service):
    self._item_service = item_service


  def to_model(self, result):
    return ItemEmb(
      id = result.ids[0],
      emb = result.embeddings[0]
    )


  def to_params(self, model):
    document = f'{model.title}:({model.release}): {model.description}'
    embedding = self._item_service.embeddings([document])[0].tolist()

    return Bunch({
        'embeddings': [embedding],
        'documents' : [document],
        'metadatas' : [{'title': model.title}],
        'ids'       : [model.id]
    })