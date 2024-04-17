import metric as mt
import util as ut
import numpy as np
from .session_step_dict import SessionStepDict
from .session import Session

@ut.printable
@ut.iterable_object
class SessionsGroup:
    def __init__(self, sessions=[]):
        self.sessions = []
        self.reset()
        [self.append(s) for s in sessions]


    def append(self, session): self.sessions.append(session if type(session) == Session else Session(session))


    def _state(self): return self.sessions


    def _elements(self): return self._state()


    @property
    def ndgc_evolution(self):
        sessions_ndgc = []

        max_steps = 0
        for session in self.sessions:
            if len(session) > max_steps:
                max_steps = len(session)

            sessions_ndgc.append(session.steps_ndcg)

        rows = []
        for values in sessions_ndgc:
            rows.append(
                np.pad(
                np.array(values),
                (0, max_steps - len(values)),
                mode            = 'constant',
                constant_values = 0
            ))

        return np.array(rows)


    @property
    def mean_mean_reciprocal_rank(self):
        return np.mean([s.mean_reciprocal_rank for s in self.sessions])


    @property
    def mean_mean_average_precision(self):
        return np.mean([s.mean_average_precision for s in self.sessions])


    @property
    def mean_mean_recall(self):
        return np.mean([s.mean_recall for s in self.sessions])


    @property
    def mean_ndgc_evolution(self):
        return ut.nanmean(self.ndgc_evolution, axis=0)

    def catalog_coverage(self, item_ids):
        recommended_item_ids = []
        for s in self.sessions:
            recommended_item_ids.extend(s.recommended_item_ids)

        return mt.catalog_coverage(recommended_item_ids, item_ids)

    @property
    def steps_by_index(self):
        groups = SessionStepDict()
        for session in self.sessions:
            for idx, step in enumerate(session):
                groups.put_step(idx+1, step)
        return groups


    @property
    def split_by_size(self):
        result = {}
        for session in self.sessions:
            sessions = result.get(len(session), SessionsGroup())
            sessions.append(session)
            result[len(session)] = sessions

        return dict(sorted(result.items()))