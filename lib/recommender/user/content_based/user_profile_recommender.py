import pandas as pd
import numpy as np
import util as ut
import recommender as rc
import pytorch_common.util as pu
import logging
import pandas as pd
import util as ut
from abc import ABCMeta, abstractmethod
from .result import UserProfileRecommenderResult


class UserProfileRecommender(rc.UserItemRecommender, metaclass=ABCMeta):
    def __init__(
        self,
        user_id_col,
        item_id_col,
        rating_col,
        emb_cols,
        col_bucket,
        unrated_items   = True,
        exclude_columns = []
    ):
        self.user_id_col     = user_id_col
        self.item_id_col     = item_id_col
        self.rating_col      = rating_col
        self.emb_cols        = emb_cols
        self._col_bucket      = col_bucket
        self._user_profile    = None
        self._item_profile    = None
        self._imdb_id_col     = 'imdb_id'
        self._unrated_items   = unrated_items
        self._exclude_columns = exclude_columns


    @property
    def user_profile(self): return self._user_profile


    @property
    def item_profile(self): return self._item_profile


    @abstractmethod
    def _train(self, df):
        pass


    def fit(self, df):
        sw = pu.Stopwatch()
        logging.info('Begin training')
        self._train(df)

        self.imdb_id_by_item_id = ut.to_dict(df, self.item_id_col, self._imdb_id_col)

        self.votes_id_by_item_id  = ut.to_dict(
            df.groupby(self.item_id_col, as_index=False)[self.rating_col].count(),
            self.item_id_col,
            self.rating_col
        )

        self.rating_by_item_id  = ut.to_dict(
            df.groupby(self.item_id_col, as_index=False)[self.rating_col].mean(),
            self.item_id_col,
            self.rating_col
        )

        self.items_by_user_id = ut.to_dict(df, self.user_id_col, self.item_id_col)

        self.user_item_df   = df[[self.user_id_col, self.item_id_col]].drop_duplicates()

        self.item_features_df = df[[self.item_id_col] + self.emb_cols].drop_duplicates(subset=[self.item_id_col])

        logging.info(f'Training finished. Time: {sw.to_str()}.')
        return self


    def user_emb(self, user_id):
        return self._user_profile.query(f'{self.user_id_col} == {user_id}')


    def _score(self, result_df, user_id):
        import warnings
        warnings.filterwarnings('ignore')

        # Add user_id, imdb_id, and rating columns
        result_df[self.user_id_col] = user_id

        result_df[self.rating_col]  = result_df[self.item_id_col].apply(lambda x: self.rating_by_item_id[x])
        result_df[self._imdb_id_col] = result_df[self.item_id_col].apply(lambda x: self.imdb_id_by_item_id[x])
        result_df['votes']           = result_df[self.item_id_col].apply(lambda x: self.votes_id_by_item_id[x])


        # Add Score columns
        target_emb_col  = list(set(self._item_profile.columns) - set([self.item_id_col]))
        # Sum of all columns score
        result_df['score'] = result_df[target_emb_col].sum(axis=1)

        result_df['popularity'] = (result_df[self.rating_col] * result_df['votes']) / (result_df[self.rating_col].max() * result_df['votes'].sum())

        result_df['raw_score'] = result_df['score']

        # result_df['score'] = (0.5 * result_df['score'] + 0.5 * result_df['popularity']) / (result_df['score'] + result_df['popularity'])


        # Filter unrated items for user
        if self._unrated_items:
            user_rated_items = self._user_rated_items(user_id)
            result_df = result_df[~result_df[self.item_id_col].isin(user_rated_items)]

        # Descendent order by score
        return result_df.sort_values(['score'], ascending=False)


    def recommend(self, user_id, k=3):
        user_emb = self.user_emb(user_id)

        if user_emb.shape[0] == 0:
            logging.warning(f'Not found user profile for {user_id} user id.')
            return self._result(
                pd.DataFrame(columns=[self.user_id_col, self.item_id_col, 'score']),
                k
            )

        result_df = self._item_profile.copy()

        for emb_col in list(set(self._item_profile.columns) - set([self.item_id_col])):
            result_df[emb_col] = result_df[emb_col].apply(lambda v: v * user_emb[emb_col].values[0] )


        result_df = result_df.merge(self.item_features_df, on=[self.item_id_col], how='left')

        return self._result(self._score(result_df, user_id), k)


    def recommend_all(self, user_ids, k=3):
        return ut.ParallelExecutor()(
            self._rec_fn,
            params          = [[u, k] for u in np.unique(user_ids)],
            fallback_result = {}
        )


    def _rec_fn(self, user_id, k):
        return self.recommend(user_id, k)

    def _result(self, df, k):
        return UserProfileRecommenderResult(
            self.__class__.__name__,
            df,
            k,
            user_id_col = self.user_id_col,
            item_id_col = self.item_id_col,
            rating_col  = self.rating_col
        )


    def _user_rated_items(self, user_id):
        return self.user_item_df[self.user_item_df[self.user_id_col] == user_id][self.item_id_col].unique()