import metric as mt
import util as ut
import numpy as np
from .session_step      import SessionStep
from .session           import Session


@ut.printable
@ut.iterable_object
class SessionStepDict:
    def __init__(self, session_by_key = {}):
        self.session_by_key = session_by_key
        self.reset()

    def put_step(self, key, step):
        session = self.session_by_key.get(key, Session())
        session.append(step if type(step) == SessionStep else SessionStep(step))
        self.put_session(key, session)
        return self

    def put_session(self, key, session):
        self.session_by_key[key] = session if type(session) == Session else Session(session)
        return self

    def __getitem__(self, key): return self.session_by_key[key]

    def filter_by_min_sessions(self, min_sessions=1):
        return SessionStepDict({stepIdx: steps for stepIdx, steps in self if len(steps) >= min_sessions})

    @property
    def keys(self): return self.session_by_key.keys()

    @property
    def sessions(self):
        from .sessions_group import SessionsGroup
        return SessionsGroup(self.session_by_key.values())

    @property
    def items(self): return self.session_by_key.items()

    @property
    def steps_count(self): return {key:len(session) for key, session in self.items}

    @property
    def mean_recall(self):
        return {stepIdex: session.mean_recall for stepIdex, session in self.items}

    @property
    def mean_average_precision(self):
        return {stepIdex: session.mean_average_precision for stepIdex, session in self.items}

    @property
    def mean_reciprocal_rank(self):
        return {stepIdex: session.mean_reciprocal_rank for stepIdex, session in self.items}


    @property
    def mean_reciprocal_rank(self):
        return {stepIdex: session.mean_reciprocal_rank for stepIdex, session in self.items}

    @property
    def mean_ndcg(self):
        return {stepIdex: session.mean_ndcg for stepIdex, session in self.items}

    @property
    def counts(self): return {stepIdex: len(session) for stepIdex, session in self }

    def _state(self): return self.session_by_key

    def _elements(self):
        return list(self.session_by_key.items())
