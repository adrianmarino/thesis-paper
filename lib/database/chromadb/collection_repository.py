from bunch import Bunch


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
        ids = df[id_col].astype(str).values.tolist()

        def get_str(row, column):
            value = row[column] if column in row else None
            return value.strip() if type(value) == str else value

        metadatas = [{c: get_str(row, c) for c in metadata_cols} for _, row in df.iterrows()] if len(
            metadata_cols) > 0 else None

        self.collection.add(
            embeddings=df[embedding_col].values.tolist() if embedding_col else [],
            documents=df[text_col].str.strip().values.tolist() if text_col else [],
            metadatas=metadatas,
            ids=ids
        )

        return self

    def search_sims(
            self,
            texts=None,
            embs=None,
            limit=5,
            where_metadata={},
            where_document={},
            include=['metadatas', 'distances']
    ):
        return self.collection.query(
            query_texts=texts,
            query_embeddings=embs,
            n_results=limit,
            where=where_metadata,
            where_document=where_document,
            include=include
        )

    def search_by_ids(self, ids, include=['embeddings', 'metadatas', 'documents']):
        return self.collection.get(ids=[str(id) for id in ids], include=include)
