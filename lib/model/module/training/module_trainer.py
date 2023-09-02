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
    def __init__(self, model, params, disable_plot=False):
        self.model  = model.to(params.model.device)
        self._params = params
        self._disable_plot = disable_plot
        ut.mkdir(self._params.model.weights_path)
        ut.mkdir(self._params.metrics.path)

    def train(self, train_ds, eval_ds):
        self.n_classes = len(train_ds.target_uniques)

        train_dl = DataLoader(
            train_ds,
            self._params.train.batch_size,
            num_workers = self._params.train.n_workers,
            pin_memory  = True,
            shuffle     = True
        )

        eval_dl  = DataLoader(
            eval_ds,
            self._params.train.batch_size,
            num_workers = self._params.train.n_workers,
            pin_memory  = True
        )

        self.model.fit(
            train_dl,
            epochs      = self._params.train.epochs,
            loss_fn     = ml.MSELossFn(),
            optimizer   = Adam(
                params = self.model.parameters(),
                lr     = self._params.train.lr
            ),
            callbacks   = [
                Validation(
                    eval_dl,
                    metrics       = { 'val_loss': ml.MSELossFn(float_result=True) },
                    each_n_epochs = 1
                ),
                ReduceLROnPlateau(
                    metric   = 'val_loss',
                    mode     = 'min',
                    factor   = self._params.train.lr_factor,
                    patience = self._params.train.lr_patience
                ),
                MetricsPlotter(
                    metrics            = ['train_loss', 'val_loss'],
                    plot_each_n_epochs = 1,
                    output_path        = f'{self._params.metrics.path}/loss',
                    disable_plot       = self._disable_plot
                ),
                Logger(['time', 'epoch', 'train_loss', 'val_loss', 'patience', 'lr']),
                SaveBestModel(
                    metric          = 'val_loss',
                    path            = ut.mkdir(self._params.model.weights_path),
                    experiment_name = self._params.metrics.experiment
                )
            ]
        )


    def evaluate(self, eval_ds):
        validator = ml.Validator(
            n_samples  = self._params.metrics.n_samples,
            batch_size = self._params.metrics.batch_size,
            metrics    = [
                mt.RMSE(),
                mt.MeanNdcgAtk            (k=5),
                mt.MeanAveragePrecisionAtk(k=5, discretizer=dr.between(4, 5)),
                mt.MeanUserFBetaScoreAtk  (k=5, n_classes=self.n_classes, discretizer=dr.between(4, 5)),
                mt.MeanUserPrecisionAtk   (k=5, n_classes=self.n_classes, discretizer=dr.between(4, 5)),
                mt.MeanUserRecallAtk      (k=5, n_classes=self.n_classes, discretizer=dr.between(4, 5))
            ],
            predictors = [ml.ModulePredictor(self.model)]
        )

        summary = validator.validate(eval_ds)
        summary.save(f'{self._params.metrics.path}/metrics')

        results = summary.show()

        summary.plot(
            log_path_builder=ut.LogPathBuilder(self._params.metrics.path),
            disable_plot=self._disable_plot
        )

        return results
