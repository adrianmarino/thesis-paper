import json
import typing
from models import Model
from mappers import ModelMapper
from .entity_already_exists_exception import EntityAlreadyExistsException
from pymongo.errors import DuplicateKeyError, BulkWriteError


class MongoRepository:
  def __init__(self, db, collection, mapper: ModelMapper, id):
    self.db = db
    self.collection = db.get_collection(collection)
    self.mapper = mapper
    self.id = id


  def add_one(self, model: Model):
    return self.add_many([model])


  def add_many(self, models: list[Model]):
    return self.add_many(models)


  async def add_many(self, models: list[Model]):
    entities = [self.mapper.to_dict(model) for model in models]
    try:
      return await self.collection.insert_many(entities)
    except (DuplicateKeyError, BulkWriteError) as e:
      raise EntityAlreadyExistsException(e)


  def update(self, model):
    properties = self.mapper.to_dict(model)
    return self.collection.update_one({self.id: properties[self.id]}, {'$set': properties})


  async def find_by_id(self, id):
    result = await self.collection.find_one({self.id: id})
    return self.mapper.to_model(result) if result else None


  async def find_one_by(self, **kwargs):
    result = await self.collection.find_one(kwargs)
    return self.mapper.to_model(result) if result else None


  async def count(self, **kwargs):
    return await self.collection.count_documents(kwargs)


  async def find_many_by(self, **kwargs):
    limit = kwargs.get('limit', None)
    kwargs.pop('limit', None)

    cursor = self.collection.find(kwargs)
    if limit:
      cursor = cursor.limit(limit)

    models = []
    for document in await cursor.to_list(length=None):
      models.append(self.mapper.to_model(document))

    return models


  def add_single_index(self, field, unique=True):
    self.add_multi_index([field], unique)


  def add_multi_index(self, fields, unique=True):
    self.collection.create_index([(f, 1) for f in fields], unique=unique)


  async def find_all(self, skip=None, limit=None):
    cursor = self.collection.find({})

    if skip:
      cursor = cursor.skip(skip)

    if limit:
      cursor = cursor.limit(limit)

    models = []
    for document in await cursor.to_list(length=None):
      models.append(self.mapper.to_model(document))

    return models


  async def delete_by_id(self, id):
    await self.collection.delete_one({self.id: id})


  async def delete_one_by(self, **kwargs):
    await self.collection.delete_one(kwargs)


  async def delete_many_by(self, **kwargs):
    await self.collection.delete_many(kwargs)
