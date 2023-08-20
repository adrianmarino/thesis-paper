from enum import Enum

import pandas as pd

import model as ml


class KNNType(Enum):
    USER_BASED = 1
    ITEM_BASED = 2


class KNNPredictorFactory:
    @staticmethod
    def create(
            train_set: pd.DataFrame,
            user_seq_col: str,
            item_seq_col: str,
            rating_col: str,
            model_type: KNNType = KNNType.USER_BASED,
            distance=ml.CosineDistance(),
            cache: bool = True
    ) -> ml.AbstractPredictor:
        if model_type == KNNType.USER_BASED:
            predictor = ml.KNNUserBasedPredictor.from_data_frame(
                data=train_set,
                user_seq_col=user_seq_col,
                item_seq_col=item_seq_col,
                rating_col=rating_col,
                distance=distance
            )
        else:
            predictor = ml.KNNItemBasedPredictor.from_data_frame(
                data=train_set,
                user_seq_col=user_seq_col,
                item_seq_col=item_seq_col,
                rating_col=rating_col,
                distance=distance
            )
        return ml.CachedPredictor(predictor) if cache else predictor
