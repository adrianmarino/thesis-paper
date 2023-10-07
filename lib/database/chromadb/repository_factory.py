from singleton_decorator import singleton
from bunch import Bunch
from database.chromadb import CollectionRepository
import chromadb
import pandas as pd
import util as ut



@singleton
class RepositoryFactory:
    def __init__(self, client=chromadb.Client()):
        self.client = client

    def create_from_cfg(self, cfgs):
        return Bunch({
            cfg.name: self.create(cfg.name, cfg.file_path, cfg.metadata_cols, cfg.embedding_col) for cfg in cfgs
        })


    def create(
            self,
            name,
            file_path,
            metadata_cols,
            embedding_col,
            id_col = 'id'
    ):
        repository = CollectionRepository(self.client, name)

        repository.insert_from_df(
            id_col        = id_col,
            df            = ut.load_df(file_path),
            metadata_cols = metadata_cols,
            embedding_col = embedding_col
        )

        return repository