import pandas as pd
import pytorch_common.util as pu
import torch
from torch.utils.data import DataLoader
import pytorch_common
import pytorch_common.util as pu
import model as ml
import util as ut
import model.training as mt


class ModulePredictionService:
    def __init__(self, model, params):
        self._model  = model.to(params.model.device)
        self._params = params

    def train(self, train_ds, test_ds):
        trainer = mt.ModuleTrainer(self._model)
        return trainer(train_ds, test_ds, self._params)

    def predict(self, df: pd.DataFrame):
        data_loader = DataLoader(
            dataset     = mt.DatasetFactory().create_from(df),
            batch_size  = self._params.train.batch_size,
            num_workers = self._params.train.n_workers
        )

        predictor = ml.ModulePredictor(self._model)

        predictions = predictor.predict_dl(data_loader)

        df[f'rating_prediction'] = predictions.numpy()

        return df