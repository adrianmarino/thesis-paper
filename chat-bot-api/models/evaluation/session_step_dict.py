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

    def mean_recall(self, min_sessions_by_step=5):
        return {step_index: Session(steps).mean_recall for step_index, steps in self.items if len(steps) >= min_sessions_by_step}

    def mean_average_precision(self, min_sessions_by_step=5):
        return {step_index: Session(steps).mean_average_precision for step_index, steps in self.items if len(steps) >= min_sessions_by_step}

    def mean_reciprocal_rank(self, min_sessions_by_step=5):
        return {step_index: Session(steps).mean_reciprocal_rank for step_index, steps in self.items if len(steps) >= min_sessions_by_step}

    def mean_ndcg(self, min_sessions_by_step=1):
        return {step_index: Session(steps).mean_ndcg for step_index, steps in self.items if len(steps) >= min_sessions_by_step}

    def _state(self): return self.steps_by_key
