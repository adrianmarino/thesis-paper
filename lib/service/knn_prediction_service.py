import numpy as np
import pandas as pd

import torch
import pytorch_common.util as pu

import model as ml
import data.dataset as ds

from data import InteractionsChangeDetector

import util as ut


class KNNPredictionService:
    def __init__(
        self,
        weights_path             : str,
        temp_path                : str,
        predictor_name           : str,
        user_seq_col             : str,
        item_seq_col             : str,
        rating_col               : str,
        distance                  = ml.CosineDistance(),
        n_neighbors              : int      = 10,
        debug                    : bool     = False,
        model_Type               : ml.KNNType = ml.KNNType.USER_BASED,
        update_period_in_minutes : int = 180, # 3 hours
        device                   = pu.get_device()
    ):
        self.weights_path    = ut.mkdir(weights_path)
        self.predictor_name  = predictor_name
        self.change_detector = InteractionsChangeDetector(
            f'{temp_path}/{predictor_name}_change_state.picket',
            user_seq_col,
            item_seq_col,
            update_period_in_minutes
        )
        self.user_seq_col    = user_seq_col
        self.item_seq_col    = item_seq_col
        self.rating_col      = rating_col
        self.distance        = distance
        self.n_neighbors     = n_neighbors
        self.debug           = debug
        self.model_Type      = model_Type
        self.predictor       = None
        self.device          = device


    def create_predictor(self, train_set: pd.DataFrame):
            predictor = ml.KNNPredictorFactory.create(
                    train_set,
                    user_seq_col = self.user_seq_col,
                    item_seq_col = self.item_seq_col,
                    rating_col   = self.rating_col,
                    model_Type   = self.model_Type,
                    distance     = self.distance
                )
            predictor.save(self.weights_path, self.predictor_name)
            return predictor

    def load_predictor(self):
        if self.model_Type == ml.KNNType.USER_BASED:
            return ml.KNNUserBasedPredictor.from_file(self.weights_path, self.predictor_name)
        else:
            return ml.KNNItemBasedPredictor.from_file(self.weights_path, self.predictor_name)


    def get_predictor(self, train_set: pd.DataFrame, force_training: bool = False):
        if force_training:
            self.predictor = self.create_predictor(train_set)
        elif self.change_detector.detect(train_set):
            self.predictor = self.create_predictor(train_set)
            self.change_detector.update(train_set)
        else:
            self.predictor = self.load_predictor()

        return self.predictor


    def to_batch(self, df):
        return torch \
            .from_numpy(df[[self.user_seq_col, self.item_seq_col]].values) \
            .to(self.device)

    def fit_predict(
        self,
        train_set      : pd.DataFrame,
        test_set       : pd.DataFrame,
        force_training : bool = False
    ):
        predictor = self.get_predictor(train_set, force_training)

        predictions = predictor.predict_batch(
            self.to_batch(test_set),
            self.n_neighbors,
            self.debug
        ).numpy()

        test_set[f'{self.rating_col}_prediction'] = predictions

        return test_set


    def delete(self):
        self.predictor.delete()