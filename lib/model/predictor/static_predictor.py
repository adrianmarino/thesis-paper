import torch
from model.predictor.abstract_predictor import AbstractPredictor
import util as ut


class StaticPredictor(AbstractPredictor):
    @staticmethod
    def from_data_frame(
      df,
      name        = None,
      rating_col  = 'rating',
      user_id_col = 'user_id',
      item_id_col = 'movie_id'
    ):
      return StaticPredictor(df, name, rating_col, user_id_col, item_id_col)


    def __init__(
      self,
      df,
      name        = None,
      rating_col  = 'rating',
      user_id_col = 'user_id',
      item_id_col = 'movie_id'
    ):
      self.index = ut.ValueIndex(df, rating_col, [user_id_col, item_id_col])
      self._name  = name


    def predict(self, user_id, item_id, n_neighbors=10, debug=False):
      return torch.tensor(self.index[[(user_id.item(), item_id.item())]])


    def predict_batch(self, batch, n_neighbors=10, debug=False):
      return torch.tensor(self.index[batch.tolist()])


    def predict_dl(self, data_loader, n_neighbors=10, debug=False):
        predictions = []

        with progress_bar(len(data_loader), f'{self.name} batch prediction') as bar:
            for features, target in data_loader:
                predictions.append(self.predict_batch(features, n_neighbors, debug))
                bar.update()

        return torch.concat(predictions)


    @property
    def name(self):
      return self._name if self._name else str(self.__class__.__name__)


    def __repr__(self): return str(self.index)


    def __str__(self):  return str(self.index)
