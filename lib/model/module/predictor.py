from abc import ABC, abstractmethod
import numpy as np
import pytorch_common.util as pu
import logging


class AbstractPredictor(ABC):
    def __init__(self, name=None): self._name = name

    @abstractmethod    
    def predict(self, user_idx, item_idx, debug=False):
        pass

    def predict_batch(self, batch, n_neighbors=10, debug=False):
        sw = pu.Stopwatch()
        prediction = np.array([self.predict(batch[i][0], batch[i][1], n_neighbors, debug) for i in range(len(batch))])
        logging.info(f'computing time: {sw.to_str()}')
        return prediction

    def evaluate(self, X, y_true, n_neighbors=10, debug=False, metrics_fn={}):
        y_pred = self.predict_batch(X, n_neighbors)
        return {name: fn(y_pred, y_true) for name, fn in metrics_fn.items()}