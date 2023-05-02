import pandas as pd
from bunch import Bunch
import util as ut


class SimItemsMixin:

    def _similar_items(self, item_ids):
        user_items_query = self._collection_repository.search_by_ids(item_ids)

        rows = []
        for user_idx in range(len(user_items_query['ids'])):
            user_item = self.__result_to_dto(user_items_query, user_idx)

            sim_items_query = self._collection_repository.search_sims(embs=[user_item.emb], limit=self.n_sim_items)
            for sim_idx in range(len(sim_items_query['ids'][0])):
                sim_item = self.__sim_result_to_dto(sim_items_query, sim_idx)

                if abs(sim_item.sim) < 1.0:
                    rows.append(self.__join_dtos(user_item, sim_item))

        return pd.DataFrame(rows)


    def __result_to_dto(self, result, idx):
        item_id = int(result['ids'][idx])
        return Bunch({
            'id': item_id,
            'imdb_id': result['metadatas'][idx]['imdb_id'],
            'title': result['metadatas'][idx]['title'],
            'emb': result['embeddings'][idx],
            'rating': self._dataset_repository.find_rating_by_item_id(item_id)
        })

    def __sim_result_to_dto(self, result, idx):
        item_id = int(result['ids'][0][idx])
        return Bunch({
            'id': item_id,
            'imdb_id': result['metadatas'][0][idx]['imdb_id'],
            'title': result['metadatas'][0][idx]['title'],
            'sim': 1 - result['distances'][0][idx],
            'rating': self._dataset_repository.find_rating_by_item_id(item_id)
        })

    def __join_dtos(self, user_item, sim_item):
        return Bunch({
            'sim': sim_item.sim,

            'sim_rating': sim_item.rating,
            'sim_id': sim_item.id,
            'sim_imdb_id': sim_item.imdb_id,
            'sim_title': sim_item.title,

            'id': user_item.id,
            'imdb_id': user_item.imdb_id,
            'title': user_item.title,
            'rating': user_item.rating
        })