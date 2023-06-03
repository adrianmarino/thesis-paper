from model.predictor.abstract_predictor import AbstractPredictor
from model import NearestNeighbors
from util import round_, delete
import logging
from data import RatingsMatrix
import torch


class KNNItemBasedPredictor(AbstractPredictor):
    @staticmethod
    def from_data_frame(data, user_seq_col, item_seq_col, rating_col, distance):
        rm = RatingsMatrix.from_dataframe(data, user_seq_col, item_seq_col, rating_col).T
        nn = NearestNeighbors(rm[:, :], distance)
        return KNNItemBasedPredictor(rm, nn)

    @staticmethod
    def from_file(path, filename):
        rm = torch.load(f'{path}/{filename}-rm.pt')
        nn = torch.load(f'{path}/{filename}-nn.pt')
        return KNNItemBasedPredictor(rm, nn)

    def __init__(self, rm, nn):
        self.rm, self.nn = rm, nn

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

    def save(self, path, filename='knn-item-predictor'):
        torch.save(self.rm, f'{path}/{filename}-rm.pt')
        torch.save(self.nn, f'{path}/{filename}-nn.pt')

    def delete(self):
        self.rm.delete()
        self.nn.delete()
        del self

