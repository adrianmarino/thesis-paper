import pandas as pd
import pytorch_common.util as pu
from bunch import Bunch

import model as ml
import util as ut
from data import InteractionsChangeDetector

params = Bunch({
    'lr': 0.001,
    'lr_factor': 0.1,
    'lr_patience': 3,
    'epochs': 25,
    'embedding_size': 50,
    'n_workers': 24,
    'batch_size': 64,
    'n_users': len(train_set.features_uniques[self._user_seq_col]),
    'n_items': len(train_set.features_uniques[self._item_seq_col])
})


class BiasGMFLoader:
    def __init__(
            self,
            model,
            trainer,
            weights_path: str,
            temp_path: str,
            predictor_name: str,
            user_seq_col: str,
            item_seq_col: str,
            update_period_in_minutes: int = 180,  # 3 hours
            device=pu.get_device()
    ):
        self._predictor_name = 'bias_gfm'
        self._weights_path = ut.mkdir(weights_path)
        self._predictor_name = predictor_name
        self._change_detector = InteractionsChangeDetector(
            f'{temp_path}/{predictor_name}_change_state.picket',
            user_seq_col,
            item_seq_col,
            update_period_in_minutes
        )
        self._device = device
        self._trainer = trainer
        self._model = model
        self._model_path = f'{self._weights_path}/{self._predictor_name}'

    def load(self, train_set: pd.DataFrame, test_set: pd.DataFrame, params):
        if self._change_detector.detect(train_set):
            self._trainer.train(self._model, train_set, test_set, params)

            self._model.save(self._model_path)

            self._change_detector.update(train_set)
        else:
            self._model.load(self._model_path)

        return model
