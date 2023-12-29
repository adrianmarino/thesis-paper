from database.chromadb import RepositoryFactory
import chromadb
import logging
from bunch import Bunch
from .repository import ChromaRepository
import os


class ChromaRepositoryFactory:
    def __init__(
      self,
      host=os.environ['CHROMA_HOST'],
      port=os.environ['CHROMA_PORT'],
    ):
      self.client = chromadb.HttpClient(host=host, port=port)
      self.factory = RepositoryFactory(self.client)


    def create(self, name, mapper):
      return ChromaRepository(self.factory.create(name), mapper)
