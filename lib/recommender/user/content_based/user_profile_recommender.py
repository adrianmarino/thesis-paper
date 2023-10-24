import pandas as pd
import numpy as np
import util as ut
import recommender as rc
import pytorch_common.util as pu
import logging
import pandas as pd
import util as ut
from abc import ABCMeta, abstractmethod


class UserProfileRecommender(rc.UserItemRecommender, metaclass=ABCMeta):
    def __init__(
        self,
        user_id_col,
        item_id_col,
        emb_cols,
        col_bucket
    ):
        self._user_id_col  = user_id_col
        self._item_id_col  = item_id_col
        self._emb_cols     = emb_cols
        self._col_bucket   = col_bucket
        self._user_profile = None
        self._item_profile = None

    @property
    def user_profile(self): return self._user_profile

    @property
    def item_profile(self): return self._item_profile

    @abstractmethod
    def fit(self, df):
        pass


    def _user_emb(self, user_id):
        return self._user_profile[self._user_profile[self._user_id_col] == user_id]


    def _score(self, result_df, user_id, sort):
        target_emb_col  = list(set(self._item_profile.columns) - set([self._item_id_col]))
        result_df['score'] = result_df[target_emb_col].sum(axis=1)
        result_df = result_df[[self._item_id_col, 'score']]
        result_df.insert(0, self._user_id_col, user_id)
        result_df = result_df[result_df['score'] > 0]
        return result_df.sort_values(['score'], ascending=False) if sort else result_df


    def recommend(self, user_id, k=10):
        user_emb = self._user_emb(user_id)

        if user_emb.shape[0] == 0:
            logging.warning(f'Not found user profile for {user_id} user id.')
            return pd.DataFrame(columns=[self._user_id_col, self._item_id_col, 'score'])

        result_df = self._item_profile.copy()
        target_emb_col  = list(set(self._item_profile.columns) - set([self._item_id_col]))
        for c in target_emb_col:
            result_df[c] = result_df[c].apply(lambda x: x *user_emb[c].values[0] )

        result_df = self._score(result_df, user_id, k is not None)

        return result_df.head(k) if k else result_df


    def recommend_all(self, user_ids, k=10):
        parallel = ut.ParallelExecutor()

        result = parallel(
            self._rec_fn,
            params          = [[u, k] for u in np.unique(user_ids)],
            fallback_result = {}
        )

        result = self._concat_user_recs(result)

        result_df = pd.DataFrame.from_dict(result)

        return result_df

    def _rec_fn(self, user_id, k):
        return self.recommend(user_id, k).to_dict('list')

    def _concat_user_recs(self, users_rec):
        result = {}

        for recs in users_rec:
            for key in recs.keys():
                if key in result:
                    result[key].extend(recs[key])
                else:
                    result[key] = recs[key]

        return result
