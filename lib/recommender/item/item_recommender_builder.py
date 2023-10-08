from .dataset_repository import DatasetRepository
from singleton_decorator import singleton
from data.dataset import MovieLensTMDBDataLoader
from database.chromadb import RepositoryFactory
from bunch import Bunch
import logging
import recommender as rc


def item_rec_sys_cfg(dataset_path, field, model):
    metadata_cols = [field, 'release_year', 'imdb_id', f'{field}_tokens']
    if field != 'title':
        metadata_cols.append('title')

    cfg = Bunch({
        'name'          : f'{field}-{model}',
        'file_path'     : f'{dataset_path}/{field}-{model}.json',
        'metadata_cols' : metadata_cols,
        'embedding_col' : f'{field}_embedding'
    })
    logging.info(f'Cfg:\n\n{cfg}')
    return cfg



@singleton
class SimilarItemRecommenderBuilder:
    def __init__(self, base_path, cfgs):
        self.repositories = RepositoryFactory().create_from_cfg(cfgs)

        self.repositories['dataset'] = DatasetRepository(
            dataset = MovieLensTMDBDataLoader.df_from_path(base_path)
        )

    def item_recommender(
        self,
        model,
        n_sim_items=10
    ):
        return rc.SimilarItemRecommender(
            self.repositories[model],
            self.repositories.dataset,
            n_sim_items
        )


    def user_item_recommender(
        self,
        model,
        n_top_rated_user_items=10,
        n_sim_items=3
    ):
        return rc.UserSimilarItemRecommender(
            self.repositories[model],
            self.repositories.dataset,
            n_top_rated_user_items,
            n_sim_items
        )