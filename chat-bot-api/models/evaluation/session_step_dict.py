import metric as mt
import util as ut
import numpy as np
from .session_step  import SessionStep
from .session       import Session


@ut.printable
class SessionStepDict:
    def __init__(self):
        self.session_by_key = {}

    def put_step(self, key, step):
        session = self.session_by_key.get(key, Session())
        session.append(step if type(step) == SessionStep else SessionStep(step))
        self.put_session(key, session)
        return self

    def put_session(self, key, session):
        self.session_by_key[key] = session if type(session) == Session else Session(step)
        return self

    def __getitem__(self, key): return self.session_by_key.get[key]

    @property
    def keys(self): return self.session_by_key.keys()

    @property
    def sessions(self): return SessionsGroup(self.session_by_key.values())

    @property
    def items(self): return self.session_by_key.items()

    @property
    def steps_count(self): return {key:len(session) for key, session in self.items}

    @property
    def step_ndgc(self):
        result = {}
        for key, session in self.items:
            if key in result:
                result[key] = np.vstack((result[key], session.steps_ndcg))
            else:
                result[key] = np.array(session.steps_ndcg)

        return dict(sorted(result.items()))


    @property
    def step_mean_ndgc(self):
        return {key: ndcgs.mean() for key, ndcgs in self.step_ndgc.items()}

    def _state(self): return self.session_by_key