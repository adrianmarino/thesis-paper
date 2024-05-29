import util as ut
import numpy as np
from .session_step import SessionStep
from .session import Session
from .sessions_group import SessionsGroup
from .session_step import SessionStep


class EvaluationState:
    @staticmethod
    def load(path):
        return ut.Picket.load(path)

    def __init__(
        self,
        recommendation_size,
        max_patience,
        plot_interval,
        profiles,
        user_ids,
        hyper_params,
        path,
    ):
        self.recommendation_size = recommendation_size
        self.max_patience = max_patience
        self.plot_interval = plot_interval
        self.profiles = profiles
        self.hyper_params = hyper_params
        self.path = path

        self.profiles_by_user_id = {u: p for p, u in zip(profiles, user_ids)}
        self.metrics_by_user_id = {}

    def get_max_patience(self, size):
        last_patience = 1
        for patience_size, patience in self.max_patience.items():
            last_patience = patience
            if size <= patience_size:
                return patience
    def save(self, path):
        ut.Picket.save(self.path, self)

    def find_profile_by_user_id(self, user_id):
        return self.profiles_by_user_id[user_id]

    def was_evaluated(self, user_id):
        return user_id in self.metrics_by_user_id

    def save_session_step(self, user_id, session_step):
        if user_id not in self.metrics_by_user_id:
            self.metrics_by_user_id[user_id] = []
        self.metrics_by_user_id.get(user_id).append(session_step)

    @property
    def sessions(self):
        return SessionsGroup(
            [Session([SessionStep(s) for s in steps]) for steps in self.metrics_by_user_id.values()]
        )