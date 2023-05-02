import recommender as rc

import chromadb
import pandas as pd
from bunch import Bunch
from singleton_decorator import singleton

import util as ut
from data.dataset import MovieLensTMDBDataLoader
from .repository_factory import RepositoryFactory


class BidirectionalHash:
    def __init__(self, df, key, value):
        self.__value_by_key = ut.to_dict(df, key, value)
        self.__key_by_value = ut.to_dict(df, value, key)

    def value_by(self, key):
        return self.__value_by_key[key] if key in self.__value_by_key else None

    def key_by(self, value):
        return self.__key_by_value[value] if value in self.__key_by_value else None


@singleton
class AppContextFactory:
    def __init__(
            self,
            base_path,
            collection_names,
            emb_path_pattern=f':PATH/movie_:NAME_embedding_bert.json'
    ):
        self.base_path = base_path
        self.collection_names = collection_names
        self.emb_path_pattern = emb_path_pattern

        repo_factory = RepositoryFactory(client=chromadb.Client())

        self.embeddings = Bunch({name: self._load(name) for name in self.collection_names})
        self.repositories = Bunch({name: repo_factory.create(name, df) for name, df in self.embeddings.items()})


        self.repositories['dataset'] = rc.DatasetRepository(
            dataset = MovieLensTMDBDataLoader.df_from_path(base_path)
        )


    def overview_item_recommender(
        self,
        n_sim_items=10
    ):
        return rc.ItemEmbDBRecommender(
            self.repositories.overview,
            self.repositories.dataset,
            n_sim_items
        )

    def genre_item_recommender(
        self,
        n_sim_items=10
    ):
        return rc.ItemEmbDBRecommender(
            self.repositories.genres,
            self.repositories.dataset,
            n_sim_items
        )

    def title_item_recommender(
        self,
        n_sim_items=10
    ):
        return rc.ItemEmbDBRecommender(
            self.repositories.title,
            self.repositories.dataset,
            n_sim_items
        )

    def tag_item_recommender(
        self,
        n_sim_items=10
    ):
        return rc.ItemEmbDBRecommender(
            self.repositories.tags,
            self.repositories.dataset,
            n_sim_items
        )

    def overview_personalized_item_recommender(
        self,
        n_top_rated_user_items=10,
        n_sim_items=3
    ):
        return rc.PersonalizedItemEmbDBRecommender(
            self.repositories.overview,
            self.repositories.dataset,
            n_top_rated_user_items,
            n_sim_items
        )

    def title_personalized_item_recommender(
        self,
        n_top_rated_user_items=10,
        n_sim_items=3
    ):
        return rc.PersonalizedItemEmbDBRecommender(
            self.repositories.title,
            self.repositories.dataset,
            n_top_rated_user_items,
            n_sim_items
        )

    def genre_personalized_item_recommender(
        self,
        n_top_rated_user_items=10,
        n_sim_items=3
    ):
        return rc.PersonalizedItemEmbDBRecommender(
            self.repositories.genres,
            self.repositories.dataset,
            n_top_rated_user_items,
            n_sim_items
        )

    def tag_personalized_item_recommender(
        self,
        n_top_rated_user_items=10,
        n_sim_items=3
    ):
        return rc.PersonalizedItemEmbDBRecommender(
            self.repositories.tags,
            self.repositories.dataset,
            n_top_rated_user_items,
            n_sim_items
        )


    def _load(self, name):
        path = self.emb_path_pattern.replace(":PATH", self.base_path).replace(":NAME", name)
        return pd.read_json(path)


