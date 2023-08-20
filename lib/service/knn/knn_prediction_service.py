import pandas as pd
import pytorch_common.util as pu
import torch


class KNNPredictionService:
    def __init__(
            self,
            predictor_loader,
            user_seq_col: str,
            item_seq_col: str,
            rating_col: str,
            device=pu.get_device()
    ):
        self._user_seq_col = user_seq_col
        self._item_seq_col = item_seq_col
        self._rating_col = rating_col
        self._device = device
        self._predictor_loader = predictor_loader

    def __call__(
            self,
            train_set: pd.DataFrame,
            test_set: pd.DataFrame,
            n_neighbors=10,
            debug=False
    ):
        predictor = self._predictor_loader.load(train_set)

        predictions = predictor.predict_batch(
            self._to_batch(test_set),
            n_neighbors,
            debug
        ).numpy()

        predictor.delete()

        test_set[f'{self._rating_col}_prediction'] = predictions

        return test_set

    def _to_batch(self, df):
        return torch \
            .from_numpy(df[[self._user_seq_col, self._item_seq_col]].values) \
            .to(self._device)
