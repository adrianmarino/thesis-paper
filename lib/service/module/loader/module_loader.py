import pandas as pd
import model as ml
import util as ut
from data import InteractionsChangeDetector
from abc import ABC, abstractmethod
import service as srv
import pytorch_common.util as pu


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
            update_period_in_min : int = 180
    ):
        self._weights_path   = ut.mkdir(weights_path)
        self._tmp_path       = ut.mkdir(tmp_path)
        self._predictor_name = predictor_name
        self._metrics_path   = ut.mkdir(metrics_path)
        self._save_file_path = f'{self._weights_path}/{self._predictor_name}'
        model_state_change_path = f'{self._save_file_path}_change_state.picket'

        self._user_seq_col = user_seq_col
        self._item_seq_col = item_seq_col
        self._rating_col   = rating_col

        self._change_detector = InteractionsChangeDetector(
            model_state_change_path,
            user_seq_col,
            item_seq_col,
            update_period_in_min
        )


    @abstractmethod
    def _create_model(self, train_set):
        pass


    def load(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        model, params = self._create_model(train_set)

        if self._change_detector.detect(train_set):
            self._train_evaluate(model, params, train_set, test_set)

            model.save(self._save_file_path)

            self._change_detector.update(train_set)

            return model, params
        else:
            model.load(self._save_file_path)
            return model, params


    def _train_evaluate(self, model, params, train_set, test_set):
        train_ds = ml.DatasetFactory().create_from(train_set)
        test_ds  = ml.DatasetFactory().create_from(test_set)

        trainer = ml.ModuleTrainer(model, params)

        trainer.train(train_ds, test_ds)
        trainer.evaluate(ml.DatasetFactory().create_from(test_set))
