from bunch import Bunch

class ChromaRepository:
  def __init__(self, repository, mapper):
    self._repository = repository
    self._mapper = mapper


  def add_one(self, model):
    params = self._mapper.to_params([model])
    self._repository.insert(params)

  def add_many(self, models):
    params = self._mapper.to_params(models)
    self._repository.insert(params)


  def find_by_id(self, id):
    result = self._repository.search_by_ids(
      ids=[id],
      include=['embeddings', 'metadatas']
    )
    return None if result.empty else self._mapper.to_model(result)


  def delete(self, id: str):
    self._repository.delete([id])


  def search_sims(self, embs, limit, where_metadata={}):
    return self._repository.search_sims(
      embs=embs,
      where_metadata=where_metadata,
      limit=limit
    )