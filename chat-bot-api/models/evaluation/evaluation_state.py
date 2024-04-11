import util as ut
import numpy as np
from .session_step  import SessionStep
from .session       import Session
from .sessions_group import SessionsGroup

from .plot import (
  smooth_lineplot,
  plot_mean_ndgc_evolution,
  plot_ndcg_sessions,
  plot_n_users_by_session_evolution_size
)
import matplotlib.pyplot as plt
import logging
import client
from faker import Faker
import pandas as pd


class EvaluationState:
    @staticmethod
    def load(path): return ut.Picket.load(path)


    def __init__(
        self,
        recomendation_size,
        max_patience,
        plot_interval,
        profiles,
        user_ids,
        hyper_params,
        path,
    ):
        self.recomendation_size  = recomendation_size
        self.max_patience        = max_patience
        self.plot_interval       = plot_interval
        self.profiles            = profiles
        self.hyper_params        = hyper_params
        self.path                = path

        self.profiles_by_user_id  = { u: p for p, u in zip(profiles, user_ids) }
        self.metrics_by_user_id  = {}

    def get_max_patience(self, size):
        last_patience = 1
        for patience_size, patience in self.max_patience.items():
            last_patience = patience
            if size <= patience_size:
                return patience
        return last_patience

    def save(self, path):
        ut.Picket.save(self.path, self)


    def find_profile_by_user_id(self, user_id):
        return self.profiles_by_user_id[user_id]


    def was_evaluated(self, user_id):
        return user_id in self.metrics_by_user_id


    def save_session(self, user_id, session):
        if user_id not in self.metrics_by_user_id:
            self.metrics_by_user_id[user_id] = []
        self.metrics_by_user_id.get(user_id).append(session)


    @property
    def session_steps_by_user(self):
        groups = SessionStepDict()
        for user_id, session in self.metrics_by_user_id.items():
            groups.put_session(user_id, Session(session))
        return groups


    def plot_metrics(self, item_ids=[]):
        plot_ndcg_sessions(
            {
                size: value.mean_ndgc_evolution
                for size, value in self.sessions.split_by_size.items()
            },
            smooth_level = 0.8,
            figsize      = (14, 6)
        )

        plot_mean_ndgc_evolution(
            self.sessions.mean_ndgc_evolution,
            smooth_level = 1,
            figsize      = (14, 6)
        )


        plot_n_users_by_session_evolution_size(
            [
                (n_steps, len(sessions))
                for n_steps, sessions in self.sessions.split_by_size.items()
            ],
            figsize      = (14, 4)
        )

        plt.show()

        logging.info(f'Mean Reciprocal Rank: {self.sessions.mean_mean_reciprocal_rank:.2}')
        logging.info(f'Mean Average Precision: {self.sessions.mean_mean_average_precision:.2}')
        logging.info(f'Recall: {self.sessions.mean_mean_recall:.2}')
        if len(item_ids) > 0:
            logging.info(f'Catalog Coverage: {self.sessions.catalog_coverage(item_ids):.2}')


    @property
    def sessions(self):
        return SessionsGroup([Session(steps) for steps in self.metrics_by_user_id.values()])