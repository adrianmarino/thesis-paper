import numpy as np
import pandas as pd
from bunch import Bunch

import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import SparseAdam, Adam
from sklearn.metrics import roc_auc_score

import pytorch_common.util as pu
from pytorch_common.modules.fn import Fn
from pytorch_common.callbacks import (
    EarlyStop,
    ReduceLROnPlateau,
    Validation,
    SaveBestModel
)
from pytorch_common.callbacks.output import Logger, MetricsPlotter

import model as ml
import data.dataset as ds

import metric as mt
import metric.discretizer as dr

import data.plot as pl
import data as dt

import logging
import random

import util as ut


class ModuleTrainer:
    def __init__(self, model):
        self._model = model

    def __call__(self, train_ds, test_ds, params):
        model = self._model.to(params.model.device)
        ut.mkdir(params.model.weights_path)
        ut.mkdir(params.metrics.path)

        train_dl = DataLoader(
            train_ds,
            params.train.batch_size,
            num_workers = params.train.n_workers,
            pin_memory  = True,
            shuffle     = True
        )

        test_dl  = DataLoader(
            test_ds,
            params.train.batch_size,
            num_workers = params.train.n_workers,
            pin_memory  = True
        )

        result = model.fit(
            train_dl,
            epochs      = params.train.epochs,
            loss_fn     = ml.MSELossFn(),
            optimizer   = Adam(
                params = model.parameters(),
                lr     = params.train.lr
            ),
            callbacks   = [
                Validation(
                    test_dl,
                    metrics       = { 'val_loss': ml.MSELossFn(float_result=True) },
                    each_n_epochs = 1
                ),
                ReduceLROnPlateau(
                    metric   = 'val_loss',
                    mode     = 'min',
                    factor   = params.train.lr_factor,
                    patience = params.train.lr_patience
                ),
                MetricsPlotter(
                    metrics            = ['train_loss', 'val_loss'],
                    plot_each_n_epochs = 1,
                    output_path        = f'{params.metrics.path}/loss'
                ),
                Logger(['time', 'epoch', 'train_loss', 'val_loss', 'patience', 'lr']),
                SaveBestModel(
                    metric          = 'val_loss',
                    path            = ut.mkdir(params.model.weights_path),
                    experiment_name = params.metrics.experiment
                )
            ]
        )

        n_classes = len(train_ds.target_uniques)

        validator = ml.Validator(
            n_samples  = params.metrics.n_samples,
            batch_size = params.metrics.batch_size,
            metrics    = [
                mt.RMSE(),
                mt.MeanNdcgAtk            (k=5),
                mt.MeanAveragePrecisionAtk(k=5, discretizer=dr.between(4, 5)),
                mt.MeanUserFBetaScoreAtk  (k=5, n_classes=n_classes, discretizer=dr.between(4, 5)),
                mt.MeanUserPrecisionAtk   (k=5, n_classes=n_classes, discretizer=dr.between(4, 5)),
                mt.MeanUserRecallAtk      (k=5, n_classes=n_classes, discretizer=dr.between(4, 5))
            ],
            predictors = [ml.ModulePredictor(model)]
        )

        summary = validator.validate(test_ds)
        summary.save(f'{params.metrics.path}/metrics')

        results = summary.show()

        summary.plot(log_path_builder=ut.LogPathBuilder(params.metrics.path))

        return results
