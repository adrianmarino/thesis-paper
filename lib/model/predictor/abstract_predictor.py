from abc import ABC, abstractmethod
import numpy as np
import pytorch_common.util as pu
import logging
import torch


class AbstractPredictor(ABC):
    @property
    def name(self): return str(self.__class__.__name__)

    @abstractmethod
    def predict(self, user_idx, item_idx, n_neighbors=10, debug=False):
        pass

    def predict_batch(self, batch, n_neighbors=10, debug=False):
        return torch.tensor([self.predict(batch[idx][0], batch[idx][1], n_neighbors, debug) for idx in range(len(batch))])


    def predict_dl(self, data_loader, n_neighbors=10, debug=False):
        predictions = []
        for features, target in data_loader:
            predictions.append(self.predict_batch(features, n_neighbors, debug))
        return torch.concat(predictions)

    def save(self, path, filename):
        raise Exception("Save functionality is not yet available!")

    def delete(self):
        raise Exception("Delete functionality is not yet available!")
