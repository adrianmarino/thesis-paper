import json
import typing
from models import Model
from .mapper import ModelMapper
from .entity_already_exists_exception import EntityAlreadyExistsException
from pymongo.errors import DuplicateKeyError, BulkWriteError


class Repository:
  def __init__(self, db, collection, mapper: ModelMapper, id):
    self.db = db
    self.collection = db.get_collection(collection)
    self.mapper = mapper
    self.id = id


  def add_one(self, model: Model):
    return self.add_many([model])


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


  def add_index(self, field, unique=True):
    self.collection.create_index([(field, 1)], unique=True)


  async def find_all(self, limit=100):
    cursor = self.collection.find({}).limit(limit)
    
    models = []
    for document in await cursor.to_list(length=10):   
      models.append(self.mapper.to_model(document))

    return models
