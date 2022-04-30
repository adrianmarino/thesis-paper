from model.module.predictor import AbstractPredictor
from model import NearestNeighbors
from util import round_
import logging
import data as dt


class KNNItemBasedPredictor(AbstractPredictor):
    @staticmethod
    def from_data_frame(data, user_seq_col, movie_seq_col, rating_col, distance):
        rm = dt.RatingsMatrix.from_dataframe(data, user_seq_col, movie_seq_col, rating_col)
        return KNNItemBasedPredictor(rm, distance)

    def __init__(self, rm, distance):
        self.rm = rm.T
        self.nn = NearestNeighbors(self.rm[:, :], distance)

 
    def predict(self, user_id, item_id, n_neighbors=10, debug=False):
        result = self.nn.neighbors(item_id, n_neighbors)
        
        numerator = 0
        total_sim = 0    
        for dist, row_idx in list(zip(result.distances, result.indexes)):
            r = self.rm[row_idx, user_id]
            if r > 0:
                sim = 1 - dist.item()

                numerator += r * sim
                total_sim += sim
        
        if total_sim == 0:
            return 0

        return numerator / total_sim

    def plot(self):
        self.rm.plot()
        self.nn.plot(prefix='Movie  ')





