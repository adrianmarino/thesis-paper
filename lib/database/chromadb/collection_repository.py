import logging
import data as dt
from .utils import to_insert_params
from bunch import Bunch
import json


class CollectionRepositorySearchByIdResult:
    def __init__(self, result = {}): self._result = result

    @property
    def ids(self): return [int(id) for id in self._result['ids']]

    @property
    def metadatas(self): return self._result['metadatas']

    @property
    def embeddings(self): return self._result['embeddings']

    @property
    def documents(self): return self._result['documents']

    @property
    def distances(self): return self._result['distances']

    def __repr__(self): return json.dumps(self._result, indent=4, sort_keys=True)

    @property
    def empty(self): return len(self.ids) == 0

    @property
    def not_empty(self): return self.empty


def take(dict, key, idx=0):
    return dict[key][idx] if key in dict and dict[key] is not None else []


class CollectionRepositorySimSearchResult(CollectionRepositorySearchByIdResult):
    def __init__(self, result = {}):
        self._result = {
            'ids'       : take(result, 'ids'),
            'metadatas' : take(result, 'metadatas'),
            'embeddings': take(result, 'embeddings'),
            'documents' : take(result, 'documents'),
            'distances' : take(result, 'distances')
        }





class CollectionRepository:
    def __init__(self, client, collection_name):
        self.client = client
        self.collection = client.get_or_create_collection(collection_name)

    def insert_from_df(
            self,
            df,
            id_col='id',
            metadata_cols=[],
            embedding_col=None,
            text_col=None
    ):
        params = to_insert_params(
            df,
            id_col,
            metadata_cols,
            embedding_col,
            text_col
        )
        return self.insert(params)


    def insert(self, params):
        non_inserted_ids = []
        with dt.progress_bar(len(params.ids), title='Insert Embeddings') as bar:
            for idx in range(len(params.ids)):
                try:
                    self.collection.add(
                        embeddings  = [params.embeddings[idx]] if params.embeddings else None,
                        metadatas   = [params.metadatas[idx]]  if params.metadatas  else None,
                        documents   = [params.documents[idx]]  if params.documents  else None,
                        ids         = [params.ids[idx]]
                    )
                except Exception as error:
                    logging.warn(f'{error}. EmbId: {params.ids[idx]}')
                    non_inserted_ids.append(params.ids[idx])
                bar.update()

        return non_inserted_ids


    def search_sims(
            self,
            texts=None,
            embs=None,
            limit=5,
            where_metadata={},
            where_document={},
            include=['metadatas', 'distances']
    ):
        return CollectionRepositorySimSearchResult(
            self.collection.query(
                query_texts=texts,
                query_embeddings=embs,
                n_results=limit,
                where=where_metadata,
                where_document=where_document,
                include=include
            )
        )

    def search_by_ids(self, ids, include=['embeddings', 'metadatas', 'documents']):
        return CollectionRepositorySearchByIdResult(self.collection.get(ids=[str(id) for id in ids], include=include))
