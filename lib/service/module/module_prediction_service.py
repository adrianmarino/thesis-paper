import pandas as pd
import pytorch_common.util as pu
import torch
from torch.utils.data import DataLoader
import pytorch_common
import pytorch_common.util as pu
import model as ml
import util as ut



class ModulePredictionService:
    def __init__(self, model_loader):
        self._model_loader = model_loader

    def predict(self, dev_set: pd.DataFrame, test_set: pd.DataFrame):
        model, params = self._model_loader.load(dev_set)

        data_loader = DataLoader(
            dataset     = ml.DatasetFactory().create_from(test_set),
            batch_size  = params.train.batch_size,
            num_workers = params.train.n_workers
        )

        predictor = ml.ModulePredictor(model)

        predictions = predictor.predict_dl(data_loader)

        test_set[f'rating_prediction'] = predictions.numpy()

        return test_set