from model.module.predictor import AbstractPredictor
from model import NearestNeighbors
from util import round_
import logging


class KNNUserBasedPredictor(AbstractPredictor):
    def __init__(self, rm, distance, name=None):
        super().__init__(name)
        self.rm = rm
        self.nn = NearestNeighbors(rm[:, :], distance)

    def predict(self, user_idx, item_idx, n_neighbors=10, debug=False):
        result = self.nn.neighbors(user_idx, n_neighbors)

        if debug:
            numerator_str = []
            total_sim_str = []

        numerator = 0
        total_sim = 0

        for dist, row_idx in list(zip(result.distances, result.indexes)):
            r = self.rm[row_idx, item_idx]
            if r > 0:
                sim = 1 - dist.item()
                deviation = self.rm.row_deviation(row_idx, item_idx)
            
                numerator += deviation * sim
                total_sim += sim

                if debug:
                    numerator_str.append(f'({deviation} * {sim})')
                    total_sim_str.append(str(sim))

        if total_sim == 0:
            return 0

        user_mean = self.rm.mean_row(user_idx)
        rating    = user_mean + (numerator / total_sim)

        if debug:
            logging.info(f'rm:\n{self.rm[:, :].cpu().numpy()}')
            logging.info(f'distances:\n{self.nn.row_distances.cpu().numpy()}')
            logging.info(f'Ratting = {user_mean} - [ ({" + ".join(numerator_str)}) / ({" + ".join(total_sim_str)}) ] = {rating}')

        return rating.item()