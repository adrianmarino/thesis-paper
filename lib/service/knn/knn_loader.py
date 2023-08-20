import pandas as pd

import model as ml
import util as ut
from data import InteractionsChangeDetector


class KNNLoader:
    def __init__(
            self,
            weights_path: str,
            temp_path: str,
            predictor_name: str,
            user_seq_col: str,
            item_seq_col: str,
            rating_col: str,
            update_period_in_minutes: int = 180,  # 3 hours,
            model_type: ml.KNNType = ml.KNNType.USER_BASED,
    ):
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
        self.model_type = model_type

    def _create_predictor(self, train_set: pd.DataFrame):
        return ml.KNNPredictorFactory.create(
            train_set,
            self._user_seq_col,
            self._item_seq_col,
            self._rating_col,
            self.model_type
        )

    def _load_predictor(self, weights_path, predictor_name):
        pred_clazz = ml.KNNUserBasedPredictor if self.model_type == ml.KNNType.USER_BASED else ml.KNNItemBasedPredictor
        return pred_clazz.from_file(weights_path, predictor_name)

    def load(self, train_set: pd.DataFrame):
        if self._change_detector.detect(train_set):
            predictor = self._create_predictor(train_set)
            predictor.save(self._weights_path, self._predictor_name)
            self._change_detector.update(train_set)
            return predictor
        else:
            return self._load_predictor(self._weights_path, self._predictor_name)
