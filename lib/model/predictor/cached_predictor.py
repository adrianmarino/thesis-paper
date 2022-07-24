from model.predictor.abstract_predictor import AbstractPredictor
import torch
import pytorch_common.util as pu


class CachedPredictor(AbstractPredictor):
    def __init__(
        self,
        predictor
    ):
        self._predictor = predictor
        self._cache     = {}

    @property
    def name(self): return self._predictor.name

    def predict(self, user_idx, item_idx, n_neighbors, debug):
        key = f'{user_idx}-{item_idx}'
        if key in self._cache:
            return self._cache[key]

        y_pred = self._predictor.predict(
            user_idx,
            item_idx,
            n_neighbors = n_neighbors,
            debug       = debug
        )

        self._cache[key] = y_pred
        return y_pred

    def _append_to_sample(self, key, value):
        if key in self._sample:
            sample = self._sample[key]
            sample = torch.add(sample, value)
        else:
            sample = torch.tensor([value], dtype=torch.float64).to(self._device)

        self._sample[key] = sample

        return sample

    def delete(self):
        self._predictor.delete()
        del self._result
        del self._sample

