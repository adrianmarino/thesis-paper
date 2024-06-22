import metric as mt
import util as ut
import numpy as np
from .session import Session
from .sessions_plotter import SessionsPlotter
import pandas as pd


@ut.printable
class SessionsGroup:
    def __init__(self, sessions = []):
        self.sessions = sessions

    def append(self, session):
        self.sessions.append(session)

    def __getitem__(self, key):
        return self.sessions[key]

    def _state(self):
        return self.sessions

    def _elements(self):
        return self._state()

    @property
    def ndgc_evolution(self):
        sessions_ndgc = []

        max_steps = 0
        for session in self.sessions:
            if len(session) > max_steps:
                max_steps = len(session)

            sessions_ndgc.append(session.ndcg)

        rows = []
        for values in sessions_ndgc:
            rows.append(
                np.pad(
                    np.array(values),
                    (0, max_steps - len(values)),
                    mode="constant",
                    constant_values=0,
                )
            )

        return np.array(rows)

    @property
    def mean_mean_reciprocal_rank(self):
        return np.mean(self.mean_reciprocal_rank)

    @property
    def mean_reciprocal_rank(self):
        return [s.mean_reciprocal_rank for s in self.sessions]

    @property
    def mean_mean_average_precision(self):
        return np.mean(self.mean_average_precision)

    @property
    def mean_average_precision(self):
        return [s.mean_average_precision for s in self.sessions]

    @property
    def step_mean_recall(self):
        return [s.mean_recall for s in self.sessions]

    @property
    def mean_recall(self):
        return np.mean(self.recall)

    @property
    def recall(self):
        return (
            np.stack([s.recall for s in self.steps])
            if len(self.steps) > 0
            else np.array([0])
        )

    @property
    def mean_ndcg(self):
        return np.mean(self.ndcg)

    @property
    def ndcg(self):
        return (
            np.stack([s.ndcg for s in self.steps])
            if len(self.steps) > 0
            else np.array([0])
        )

    @property
    def steps(self):
        return [step for session in self.sessions for step in session.steps]

    @property
    def steps_mean_ndcg(self):
        return ut.nanmean(self.ndgc_evolution, axis=0)

    def catalog_coverage(self, item_ids):
        recommended_item_ids = []
        for s in self.sessions:
            recommended_item_ids.extend(s.recommended_item_ids)

        return mt.catalog_coverage(recommended_item_ids, item_ids)

    @property
    def steps_by_index(self):
        from .session_step_dict import SessionStepDict
        session_steps = SessionStepDict()
        for session in self.sessions:
            for idx, step in enumerate(session.steps):
                session_steps.put_step(idx + 1, step)
        return session_steps

    @property
    def group_by_steps_count(self):
        result = {}
        for session in self.sessions:
            key = len(session.steps)
            sessions = result.get(key, SessionsGroup())
            sessions.append(session)
            result[key] = sessions

        return dict(sorted(result.items()))

    @property
    def n_found_relevant_items(self):
        return np.array([len(step.found_relevant_items) for step in self.steps])


    @property
    def n_found_relevant_items_by_step_index(self):
        data = []
        for step_index, steps in self.steps_by_index.items:
            for step in steps:
                data.append({
                    'step_index': step_index,
                    'n_found_relevant_items': len(step.found_relevant_items)
                })
        return pd.DataFrame(data)

    @property
    def steps(self):
        return np.array([step for session in self.sessions for step in session.steps]).flatten()

    @property
    def plotter(self): return SessionsPlotter(self)


    def n_session_with_more_than(self, n_items=30):
        df = pd.DataFrame(
            [(idx, len(s.found_relevant_items)) for idx, s in enumerate(self.sessions)],
            columns=['session', 'found_relevant_items'],
        ).pipe(ut.group_size, 'found_relevant_items')
        df = df.rename(columns={'size': 'n_sessions', 'found_relevant_items': 'n_rated_items'})

        df = df[df['n_rated_items'] >= n_items]

        return df['n_sessions'].sum()


    def __len__(self):
        return len(self.sessions)