import pandas as pd
import model as ml
import util as ut
from data import InteractionsChangeDetector
from abc import ABC, abstractmethod
import service as srv
import pytorch_common.util as pu
from sklearn.model_selection import train_test_split
import logging
import glob
import os


class ModuleLoader(ABC):
    def __init__(
            self,
            weights_path         : str,
            metrics_path         : str,
            tmp_path             : str,
            predictor_name       : str,
            user_seq_col         : str = 'user_seq',
            item_seq_col         : str = 'item_seq',
            rating_col           : str = 'rating',
            update_period_in_min : int = 180,
            disable_plot=True
    ):
        self._weights_path   = ut.mkdir(weights_path)
        self._tmp_path       = ut.mkdir(tmp_path)
        self._predictor_name = predictor_name
        self._metrics_path   = ut.mkdir(metrics_path)
        self._save_file_path = f'{self._weights_path}/{self._predictor_name}'
        model_state_change_path = f'{self._tmp_path}/{self._predictor_name}_change_state.picket'

        self._user_seq_col = user_seq_col
        self._item_seq_col = item_seq_col
        self._rating_col   = rating_col

        self._change_detector = InteractionsChangeDetector(
            model_state_change_path,
            user_seq_col,
            item_seq_col,
            update_period_in_min
        )
        self._disable_plot = disable_plot


    @abstractmethod
    def _create_model(self, train_set):
        pass


    def load(self, dev_set: pd.DataFrame):
        model, params = self._create_model(dev_set)

        if self._change_detector.detect(dev_set):
            self._train_evaluate(model, params, dev_set)

            model.save(self._save_file_path)

            self._change_detector.update(dev_set)

            return model, params
        else:
            model.load(self._save_file_path)
            return model, params


    def _train_evaluate(self, model, params, dev_set):
        [os.remove(path) for path in glob.glob(f'{self._weights_path}/*{self._predictor_name}*')]
        ut.remove_dir(self._metrics_path)

        trainer = ml.ModuleTrainer(model, params, disable_plot=self._disable_plot)

        train_set, eval_set = train_test_split(
            dev_set,
            shuffle=True,
            test_size=params.train.eval_percent
        )
        logging.info(f'Train: {train_set.shape[0]}, Eval: {eval_set.shape[0]}')

        train_ds = ml.DatasetFactory().create_from(train_set)
        eval_ds  = ml.DatasetFactory().create_from(eval_set)

        trainer.train(train_ds, eval_ds)
        trainer.evaluate(eval_ds)
