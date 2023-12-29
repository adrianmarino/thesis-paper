from database.chromadb import RepositoryFactory
import chromadb
import logging
from bunch import Bunch
from .repository import ChromaRepository


class ChromaRepositoryFactory:
    def __init__(self):
      self.client = chromadb.HttpClient(host='0.0.0.0', port=9090)
      self.factory = RepositoryFactory(self.client)


    def create(self, name, mapper):
      return ChromaRepository(self.factory.create(name), mapper)
