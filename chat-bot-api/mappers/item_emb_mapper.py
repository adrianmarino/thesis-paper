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


  def to_params(self, models):
    documents = []
    metadatas = []
    ids = []

    for model in models:
      documents.append(f'{model.title}:({model.release}): {model.description}')
      metadatas.append({'title': model.title})
      ids.append(model.id)

    embeddings = self._item_service.embeddings(documents).tolist()

    return Bunch({
        'embeddings': embeddings,
        'documents' : documents,
        'metadatas' : metadatas,
        'ids'       : ids
    })