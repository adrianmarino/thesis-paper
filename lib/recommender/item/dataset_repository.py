import pandas as pd
import util as ut

def group_mean(df, group_col, mean_col):
    return df.groupby([group_col])[mean_col].mean().reset_index()


def mean_by_key(df, key, value):
    return ut.to_dict(group_mean(df, key, value), key, value)


class DatasetRepository:
    def __init__(self, dataset):
        self.__dataset = dataset
        self.__ratings_by_item_id = mean_by_key(self.__dataset, 'movie_id', 'user_movie_rating')

    def find_rating_by_item_id(self, item_id):
        return self.__ratings_by_item_id[item_id] if item_id in self.__ratings_by_item_id else None

    def find_top_rated_item_by_user_id(self, user_id, k):
        user_items = self.__dataset[self.__dataset['user_id'].isin([user_id])].copy()


        user_items['score'] = user_items['movie_id'].apply(lambda id: self.find_rating_by_item_id(id))

        return user_items[['movie_id', 'score']] \
            .drop_duplicates()['score'] \
            .nlargest(n=k)

    def users_id_from_movie_id(self, movie_id):
        return self.__dataset[self.__dataset['movie_id'] == movie_id]['user_id'].unique()