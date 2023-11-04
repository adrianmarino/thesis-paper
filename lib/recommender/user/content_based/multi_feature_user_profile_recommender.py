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
        user_id_col     = 'user_id',
        item_id_col     = 'movie_id',
        rating_col      = 'rating',
        emb_cols        = [ 'genres', 'adults', 'language', 'year' ],
        col_bucket      = { 'year': 10 },
        exclude_columns = [],
        unrated_items   = True
    ):
        super().__init__(
            user_id_col,
            item_id_col,
            rating_col,
            emb_cols,
            col_bucket,
            unrated_items,
            exclude_columns
        )

    @property
    def name(self):
        return f'UserProfileRecommender({", ".join(self.emb_cols)})'


    def _train(self, df):
        cols = [self.user_id_col, self.item_id_col] + self.emb_cols

        one_hot_df = ut.one_hot(df[cols], self.emb_cols, self._col_bucket)

        self._user_profile = self._build_user_profile(df, one_hot_df)

        self._item_profile = self._build_item_profile(one_hot_df)

        return self


    def _build_user_profile(self, df, one_hot_df):
        # Add rating
        one_hot_df2 = one_hot_df.copy()
        one_hot_df2[self.rating_col] = df[self.rating_col]

        # Multiply emb_cols by rating
        tmp_df = ut.multiply_by(one_hot_df2, self._target_emb_cols(one_hot_df), self.rating_col)
        # Get total
        total = tmp_df.apply(lambda row: row.sum(), axis=1).sum()

        tmp_df[self.user_id_col] = one_hot_df2[self.user_id_col]

        # Sum all emb_cols by user
        tmp_df2 = ut.group_sum(tmp_df, self.user_id_col)

        # Sum normalize emb_cols by total
        tmp_df3 = tmp_df2[ut.subtract(tmp_df2.columns, [self.user_id_col])] / total
        tmp_df3[self.user_id_col] = tmp_df2[self.user_id_col]

        return tmp_df3.drop(columns=self._exclude_columns, errors='ignore')


    def _build_item_profile(self, one_hot_df):
        return one_hot_df[self._target_emb_cols(one_hot_df) + [self.item_id_col]] \
            .drop_duplicates(subset=[self.item_id_col]) \
            .drop(columns=self._exclude_columns, errors='ignore')


    def _target_emb_cols(self, df):
        non_emb_cols = [self.user_id_col, self.item_id_col, self.rating_col] + self.emb_cols

        return ut.subtract(df.columns, non_emb_cols)