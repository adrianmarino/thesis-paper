from abc import ABC, abstractmethod
import numpy as np
import pytorch_common.util as pu
import logging
import torch
from data import progress_bar
import pandas as pd


class AbstractPredictor(ABC):
    @property
    def name(self): return str(self.__class__.__name__)


    @abstractmethod
    def predict(self, user_idx, item_idx, n_neighbors=10, debug=False):
        pass


    def predict_batch(self, batch, n_neighbors=10, debug=False):
        results = []

        if len(batch) <= 1:
            results.append(self.predict(batch[0][0], batch[0][1], n_neighbors, debug))
        else:
            with progress_bar(len(batch), f'{self.name} prediction') as bar:
                for idx in range(len(batch)):
                    results.append(self.predict(batch[idx][0], batch[idx][1], n_neighbors, debug))
                    bar.update()

        return torch.tensor(results)

    def predict_dl(self, data_loader, n_neighbors=10, debug=False):
        predictions = []

        with progress_bar(len(data_loader), f'{self.name} batch prediction') as bar:
            for features, target in data_loader:
                predictions.append(self.predict_batch(features, n_neighbors, debug))
                bar.update()

        return torch.concat(predictions)


    def save(self, path, filename):
        raise Exception("Save functionality is not yet available!")


    def delete(self):
        raise Exception("Delete functionality is not yet available!")
