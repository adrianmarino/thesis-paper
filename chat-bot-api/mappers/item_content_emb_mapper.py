from .mapper      import ModelMapper
from models       import EntityEmb
from bunch import Bunch
from sentence_transformers import SentenceTransformer

class ItemContentEmbMapper(ModelMapper):
  def __init__(self, item_service):
    self._item_service = item_service


  def to_model(self, result):
    return EntityEmb(
      id = result.ids[0],
      emb = result.embeddings[0]
    )


  def to_params(self, models):
    documents = []
    metadatas = []
    ids = []

    for model in models:
      documents.append(f'Title: {model.title}. Description: {model.description}. Genres: {", ".join(model.genres)}.')
      metadatas.append({'id': model.id, 'release': int(model.release), 'genres': ','.join(model.genres)})
      ids.append(model.id)

    embeddings = self._item_service.generate(documents)

    return Bunch({
        'embeddings': embeddings.tolist(),
        'documents' : documents,
        'metadatas' : metadatas,
        'ids'       : ids
    })