from singleton_decorator import singleton

from database.chromadb import CollectionRepository


@singleton
class RepositoryFactory:
    def __init__(self, client):
        self.client = client

    def create(
            self,
            name,
            df,
            metadata_cols=[
                'release_year',
                'title',
                'imdb_id'
            ]
    ):
        return CollectionRepository(self.client, name).insert_from_df(
            df,
            metadata_cols=[f'{name}_tokens'] + metadata_cols,
            embedding_col=f'{name}_embedding'
        )
