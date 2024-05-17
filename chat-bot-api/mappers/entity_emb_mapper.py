from .mapper      import ModelMapper
from models       import EntityEmb
from bunch import Bunch
from sentence_transformers import SentenceTransformer

class EntityEmbMapper(ModelMapper):
  def to_model(self, result):
    return EntityEmb(
      id = str(result.ids[0]),
      emb = result.embeddings[0]
    )

  def to_params(self, models):
    ids = []
    embeddings = []

    for model in models:
      ids.append(model.id)
      embeddings.append(model.emb)

    return Bunch({
        'embeddings': embeddings,
        'ids'       : ids,
        'metadatas' : [],
        'documents' : []
    })