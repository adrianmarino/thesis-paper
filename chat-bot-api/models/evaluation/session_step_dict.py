import metric as mt
import util as ut
import numpy as np
from .session_step      import SessionStep
from .session           import Session


@ut.printable
class SessionStepDict:
    def __init__(self):
        self.steps_by_key = {}

    def put_step(self, key, step):
        steps = self.steps_by_key.get(key, [])
        steps.append(step)
        self.steps_by_key[key] = steps
        return self

    def __getitem__(self, key): return self.steps_by_key[key]

    def filter_by_min_sessions(self, min_sessions=1):
        return SessionStepDict({key: steps for key, steps in self.items if len(steps) >= min_sessions})

    @property
    def items(self): return self.steps_by_key.items()

    @property
    def mean_recall(self):
        return {key: Session(steps).mean_recall for key, steps in self.items}

    @property
    def mean_average_precision(self):
        return {key: Session(steps).mean_average_precision for key, steps in self.items}

    @property
    def mean_reciprocal_rank(self):
        return {key: Session(steps).mean_reciprocal_rank for key, steps in self.items}


    @property
    def mean_reciprocal_rank(self):
        return {key: Session(steps).mean_reciprocal_rank for key, steps in self.items}

    @property
    def mean_ndcg(self):
        return {key: Session(steps).mean_ndcg for key, steps in self.items}

    def _state(self): return self.steps_by_key
