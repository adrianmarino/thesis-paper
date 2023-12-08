import torch
from model.predictor.abstract_predictor import AbstractPredictor
import util as ut
import random


class RandomPredictor(AbstractPredictor):
    def __init__(self, df, rating_col):
      self.ratings = df[rating_col].unique()


    def predict(self, user_id, item_id, n_neighbors=10, debug=False):
      return torch.tensor(random.choice(self.ratings))


    def predict_batch(self, batch, n_neighbors=10, debug=False):
      return torch.tensor([random.choice(self.ratings) for i, b in enumerate(batch)])


    def predict_dl(self, data_loader, n_neighbors=10, debug=False):
        predictions = []


        with progress_bar(len(data_loader), f'{self.name} batch prediction') as bar:
            for features, target in data_loader:
                predictions.append(self.predict_batch(features, n_neighbors, debug))
                bar.update()

        return torch.concat(predictions)
