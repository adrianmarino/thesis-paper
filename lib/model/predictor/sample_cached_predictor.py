from model.predictor.abstract_predictor import AbstractPredictor
import torch
import pytorch_common.util as pu


class SampleCachedPredictor(AbstractPredictor):
    def __init__(
        self,
        predictor,
        sample_size = 10,
        reduce_fn   = lambda x: torch.mean(x),
        device      = pu.get_device()
    ):
        self._predictor   = predictor
        self._result      = {}
        self._sample      = {}
        self._reduce_fn   = reduce_fn
        self._sample_size = sample_size
        self._device      = device

    @property
    def name(self): return self._predictor.name

    def predict(self, user_idx, item_idx, n_neighbors, debug):
        key = f'{user_idx}-{item_idx}'
        if key in self._result:
            return self._result[key]

        y_pred = self._predictor.predict(
            user_idx,
            item_idx,
            n_neighbors = n_neighbors,
            debug       = debug
        )

        sample         = self._append_to_sample(key, y_pred)
        sampled_y_pred = self._reduce_fn(sample)

        if len(sample) >= self._sample_size:
            self._result[key] = sampled_y_pred

        return sampled_y_pred

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

