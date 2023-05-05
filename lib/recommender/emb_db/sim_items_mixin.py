import pandas as pd
from bunch import Bunch
import util as ut
import logging


class SimItemsMixin:

    def _similar_items(self, item_ids):
        source_items_query = self._collection_repository.search_by_ids(item_ids)

        logging.info(f'Found {len(source_items_query["ids"])} items by ids: {item_ids}.')

        rows = []
        for source_item_idx in range(len(source_items_query['ids'])):
            source_item = self.__result_to_dto(source_items_query, source_item_idx)

            sim_items_query = self._collection_repository.search_sims(embs=[source_item.emb], limit=self.n_sim_items)

            logging.info(f'Found {len(sim_items_query["ids"][0])} similar to {source_item.id} item.')

            for sim_idx in range(len(sim_items_query['ids'][0])):
                sim_item = self.__sim_result_to_dto(sim_items_query, sim_idx)

                if sim_item.id not in item_ids:
                    rows.append(self.__join_dtos(source_item, sim_item))


        return pd.DataFrame(rows)


    def __result_to_dto(self, result, idx):
        item_id = int(result['ids'][idx])
        return Bunch({
            'id'      : item_id,
            'imdb_id' : result['metadatas'][idx]['imdb_id'],
            'title'   : result['metadatas'][idx]['title'],
            'emb'     : result['embeddings'][idx],
            'rating'  : self._dataset_repository.find_rating_by_item_id(item_id)
        })

    def __sim_result_to_dto(self, result, idx):
        item_id = int(result['ids'][0][idx])
        return Bunch({
            'id'      : item_id,
            'imdb_id' : result['metadatas'][0][idx]['imdb_id'],
            'title'   : result['metadatas'][0][idx]['title'],
            'sim'     : 1 - result['distances'][0][idx],
            'rating'  : self._dataset_repository.find_rating_by_item_id(item_id)
        })

    def __join_dtos(self, source_item, sim_item):
        return Bunch({
            'sim': sim_item.sim,

            'sim_rating' : sim_item.rating,
            'sim_id'     : sim_item.id,
            'sim_imdb_id': sim_item.imdb_id,
            'sim_title'  : sim_item.title,

            'id'      : source_item.id,
            'imdb_id' : source_item.imdb_id,
            'title'   : source_item.title,
            'rating'  : source_item.rating
        })