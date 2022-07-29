import pandas as pd
from .recommender import Recommender
import data as dt
from .result.impl.single_recommender_result import SingleRecommenderResult

def weighted_mean(df, weights):
    total = [df[f'{column}_distance'] * weight for column, weight in weights.items()]
    return sum(total) / len(weights.items())


class EnsembleWeightedDistanceMatrixRecommender(Recommender):
    def __init__(self, recommenders, weights):
        self.__recommenders = recommenders
        self.__weights      = weights
        self.df             = recommenders[0].df

    def recommend(self, item_index, user_id=None, k=5):
        if k != None:
            k +=1

        rec_dfs = {r.column: r.recommend(item_index, k=None).recommendations for r in self.__recommenders}

        result = pd.DataFrame()
        for column, rec_df in rec_dfs.items():
            rec_df = rec_df.rename(columns={'distance': f'{column}_distance'})
            if result.empty:
                result = rec_df
            else:
                rec_df = rec_df.drop(columns=['index', 'title', 'imdb_id'])
                result = pd.merge(result, rec_df, on='id')

        result['distance'] = weighted_mean(result, self.__weights)
        result = result[['distance', 'title', 'imdb_id']]

        name = [r.column for r in self.__recommenders]
        item = self.df.iloc[[ item_index]][['id', 'title', 'imdb_id']]
        recommendations = result.sort_values(by=['distance']).pipe(dt.reset_index)[1:k]

        return SingleRecommenderResult(name, item, recommendations)