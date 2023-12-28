from database.chromadb import RepositoryFactory
import logging
from bunch import Bunch
from .repository import ChromaRepository


class ChromaRepositoryFactory:
    @staticmethod
    def create(name, mapper):
      return ChromaRepository(
        RepositoryFactory().create(name),
        mapper
      )



