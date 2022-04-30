from abc import ABC, abstractmethod
import numpy as np
import pytorch_common.util as pu
import logging


class AbstractPredictor(ABC):
    @property
    def name(self): return str(self.__class__.__name__)

    @abstractmethod    
    def predict(self, user_idx, item_idx, debug=False):
        pass

    def predict_batch(self, batch, n_neighbors=10, debug=False):
        sw = pu.Stopwatch()
        prediction = np.array([self.predict(batch[i][0], batch[i][1], n_neighbors, debug) for i in range(len(batch))])
        logging.debug(f'computing time: {sw.to_str()}')
        return prediction