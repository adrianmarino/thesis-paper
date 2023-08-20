

port pandas as pd
import pytorch_common.util as pu
from bunch import Bunch
from pytorch_common.callbacks import (ReduceLROnPlateau, Validation)
from pytorch_common.callbacks.output import Logger
from torch.optim import Adam
from torch.utils.data import DataLoader

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
            weights_path: str,
            temp_path: str,
            predictor_name: str,
            user_seq_col: str,
            item_seq_col: str,
            rating_col: str,
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
        self._user_seq_col = user_seq_col
        self._item_seq_col = item_seq_col
        self._rating_col = rating_col
        self._device = device

    def _train(
            self,
            train_set: pd.DataFrame,
            eval_set: pd.DataFrame
    ):


        train_dl = DataLoader(
            train_set,
            params.batch_size,
            num_workers=params.n_workers,
            pin_memory=True,
            shuffle=True
        )

        eval_dl = DataLoader(
            eval_set,
            params.batch_size,
            num_workers=params.n_workers,
            pin_memory=True,
            shuffle=True
        )

        model = ml.BiasedGMF(
            params.n_users,
            params.n_items,
            params.embedding_size
        ).to(pu.get_device())

        model.fit(
            train_dl,
            epochs=params.epochs,
            loss_fn=ml.MSELossFn(),
            optimizer=Adam(
                params=model.parameters(),
                lr=params.lr
            ),
            callbacks=[
                Validation(
                    eval_dl,
                    metrics={'val_loss': ml.MSELossFn(float_result=True)},
                    each_n_epochs=1
                ),
                ReduceLROnPlateau(
                    metric='val_loss',
                    mode='min',
                    factor=params.lr_factor,
                    patience=params.lr_patience
                ),
                Logger(['time', 'epoch', 'train_loss', 'val_loss', 'patience'])
            ]
        )

        return model

    def load(self, train_set: pd.DataFrame, test_set: pd.DataFrame, params):
        model = ml.BiasedGMF(
                params.n_users,
                params.n_items,
                params.embedding_size
            ).to(pu.get_device())

        if self._change_detector.detect(train_set):
            predictor = self._train(model, train_set, test_set, params)
            predictor.save(self._weights_path)
            self._change_detector.update(train_set)
            return predictor
        else:

            model.load(f'{self._weights_path}/{self._predictor_name}')

        return model