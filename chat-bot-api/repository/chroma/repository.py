from bunch import Bunch

class ChromaRepository:
  def __init__(self, repository, mapper):
    self._repository = repository
    self._mapper = mapper


  def upsert_one(self, model):
    params = self._mapper.to_params([model])
    self._repository.insert(params)

  def upsert_many(self, models):
    params = self._mapper.to_params(models)
    self._repository.insert(params)


  def find_by_id(self, id):
    result = self._repository.search_by_ids(
      ids=[id],
      include=['embeddings', 'metadatas']
    )
    return None if result.empty else self._mapper.to_model(result)


  def delete(self, id: str):
    self.delete_many([id])


  def delete_many(self, ids: list[str]):
    self._repository.delete(ids)


  def search_sims(self, embs, limit, where_metadata={}, where_document={}):
    return self._repository.search_sims(
      embs=embs,
      where_metadata=where_metadata,
      where_document=where_document,
      limit=limit
    )


  def count(self):
    return self._repository.count()