import logging
import recommender as rc
import util as ut


def subtract(a, b):
    return list(set(a) - set(b))


def group_count(df, group_col):
    return df.groupby([group_col]).sum().reset_index()


class MultiFeatureUserProfileRecommender(rc.UserProfileRecommender):
    def __init__(
        self,
        user_id_col  = 'user_id',
        item_id_col  = 'movie_id',
        emb_cols     = [ 'genres', 'adults', 'language', 'year' ],
        col_bucket   = { 'year': 10 }
    ):
        super().__init__(user_id_col, item_id_col, emb_cols, col_bucket)


    def fit(self, df):
        cols = [self._user_id_col, self._item_id_col] + self._emb_cols

        df = df[cols]

        one_hot_df = ut.one_hot(df, self._emb_cols, self._col_bucket)

        self._user_profile = one_hot_df[subtract(one_hot_df.columns, self._emb_cols + [self._item_id_col])]

        self._user_profile = group_count(self._user_profile, self._user_id_col)

        emb_cols = subtract(self._user_profile.columns, [self._user_id_col])

        emb_df = self._user_profile[emb_cols]
        emb_df = emb_df.apply(lambda row: row / row.sum(), axis=1)
        emb_df = emb_df.dropna()
        emb_df.insert(0, self._user_id_col, self._user_profile[self._user_id_col])
        self._user_profile = emb_df

        self._item_profile = one_hot_df[subtract(one_hot_df.columns, self._emb_cols + [self._user_id_col])]
        self._item_profile = self._item_profile.drop_duplicates(subset=[self._item_id_col])

        return self