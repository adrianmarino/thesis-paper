import metric as mt
import util as ut
import numpy as np
from .session_step import SessionStep


@ut.printable
@ut.iterable_object
class Session:
    def __init__(self, steps=[]):
        self.steps = steps

    def append(self, step):
        self.steps.append(step)
        return self

    def __getitem__(self, idx):
        return self.steps[idx]

    @property
    def mean_reciprocal_rank(self):
        return mt.mean_reciprocal_rank(
            [s.recommended_item_ids for s in self.steps],
            [s.relevant_item_ids for s in self.steps],
        )

    @property
    def mean_average_precision(self):
        return mt.mean_average_precision(
            [s.recommended_item_ids for s in self.steps],
            [s.relevant_item_ids for s in self.steps],
        )

    @property
    def mean_recall(self):
        return np.mean(self.recall)

    @property
    def recall(self):
        return [s.recall for s in self.steps]

    @property
    def mean_ndcg(self):
        return np.mean(self.ndcg)

    @property
    def ndcg(self):
        return [s.ndcg for s in self.steps]

    @property
    def recommended_item_ids(self):
        return [c.recommended_item_ids for c in self.steps]

    def catalog_coverage(self, item_ids):
        return mt.catalog_coverage(self.recommended_item_ids, item_ids)

    def _state(self):
        return self.steps

    def _elements(self):
        return self._state()
