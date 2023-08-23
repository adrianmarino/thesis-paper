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
        self._model   = model.to(params.model.device)
        self._trainer = mt.ModuleTrainer(model, params)
        self._params  = params
        self._summary = None

    def train(self, train_set, test_set):
        if self._summary:
            return self._summary

        train_ds = mt.DatasetFactory().create_from(train_set)
        test_ds  = mt.DatasetFactory().create_from(test_set)

        self._summary = self._trainer.train(train_ds, test_ds)
        return self._summary


    def evaluate(self, df):
        ds = mt.DatasetFactory().create_from(df)
        return self._trainer.evaluate(ds)


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


    def train_evaluate_predict(self, train_set, test_set):
        self.train(train_set, test_set)
        summary = self.evaluate(test_set)
        self.predict(test_set)
        return summary