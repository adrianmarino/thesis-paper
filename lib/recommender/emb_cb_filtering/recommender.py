from .result import EmbCBFilteringRecommenderResult
import pandas as pd


class EmbCBFilteringRecommender:
    def __init__(
        self,
        field,
        emb_repository,
        dataset,
        user_id_col      = 'user_id',
        item_id_col      = 'movie_id',
        rating_col       = 'user_movie_rating',
        imdb_id_col      = 'movie_imdb_id',
        release_year_col = 'movie_release_year',
        metadata         = ['movie_genres']
    ):
        self.__name = f'{field}-cb-recommender'
        self.__emb_repository   = emb_repository
        self.__dataset          = dataset
        self.__user_id_col      = user_id_col
        self.__item_id_col      = item_id_col
        self.__rating_col       = rating_col
        self.__imdb_id_col      = imdb_id_col
        self.__release_year_col = release_year_col
        self.__metadata         = metadata


    def __find_similar_users_by_id(self, user_id, k):
        result = self.__emb_repository.search_by_ids([user_id])

        if result.empty:
            return []

        similar_users = self.__emb_repository.search_sims(
            embs=[result.embeddings[0]],
            limit=k
        )

        return similar_users

    def __similar_users_items(self, ids):
        return self.__dataset.data[self.__dataset.data[self.__user_id_col].isin(ids)]


    def __score(self, df, similar_users_result):
        user_distances = {id: similar_users_result.distances[idx] for idx, id in enumerate(similar_users_result.ids)}

        recommendations = df.copy()
        recommendations['score'] = recommendations[self.__user_id_col] \
            .apply(lambda r:  (1 - user_distances[r]))  * df[self.__rating_col]

        return recommendations.groupby([self.__item_id_col, self.__release_year_col])['score'] \
            .mean() \
            .reset_index() \
            .sort_values(['score', self.__release_year_col], ascending=False)

    def __create_result(self, items_df, k):
        items_df = items_df.drop([self.__release_year_col], axis=1)

        df = items_df.merge(self.__dataset.data, on=self.__item_id_col)
        df = df[['score', self.__item_id_col, self.__imdb_id_col, self.__release_year_col]+self.__metadata]
        df = df.drop_duplicates(subset=[self.__item_id_col]).head(k)

        return EmbCBFilteringRecommenderResult(self.name, df, self.__imdb_id_col, 'score', self.__metadata + [self.__release_year_col])


    def recommend(self, user_id: int = None, k: int = 5, k_sim_users: int = 15):
        similar_users_result = self.__find_similar_users_by_id(user_id, k_sim_users)

        similar_users_items_df =  self.__similar_users_items(similar_users_result.ids)

        scored_items = self.__score(similar_users_items_df, similar_users_result)

        return self.__create_result(scored_items, k)


    @property
    def name(self):
        return self.__name