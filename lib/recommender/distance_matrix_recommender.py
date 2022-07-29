import pytorch_common.util as pu
import model as ml
import torch
import numpy as np
from .recommender import Recommender
from .result.impl.single_recommender_result import SingleRecommenderResult
import data as dt


def emb_as_dist_matrix(df, column, distance_fn=ml.CosineDistance(), device=pu.get_device()):    
    values = np.array([np.array(c, dtype=np.float) for c in df[column].values])
    embedding = torch.tensor(values).to(device)
    return ml.rows_distance_matrix(embedding, distance_fn, device=device)


class DistanceMatrixRecommender(Recommender):
    def __init__(self, df, column, device=pu.get_device()):
        self.dist_matrix = emb_as_dist_matrix(df, column, device=device)
        self.df        = df.pipe(dt.reset_index)
        self.column    = column.split("_embedding")[0]

    def recommend(self, item_index, user_index=None, k=5):
        if k:
            k +=1

        similar_movies = self.dist_matrix[item_index, :].cpu().numpy()
        similar_indexes = np.argsort(similar_movies)[:k]

        recommendations = self.df.copy()
        recommendations['distance'] = 0
        for idx in similar_indexes:
           recommendations.iloc[idx, recommendations.columns.get_loc('distance')] = similar_movies[idx]
        recommendations = recommendations.iloc[similar_indexes][['distance', 'id', 'title', 'imdb_id']]

        return SingleRecommenderResult(
            self.column,
            self.df.iloc[[item_index]][['id', 'title', 'imdb_id']],
            recommendations.reset_index()[1:k]
        )