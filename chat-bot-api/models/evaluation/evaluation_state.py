import util as ut
import numpy as np
from .session_step import SessionStep
from .session import Session
from .sessions_group import SessionsGroup

from .plot import (
    smooth_lineplot,
    plot_smooth_line,
    plot_ndcg_sessions,
    plot_n_users_by_session_evolution_size,
)
import matplotlib.pyplot as plt
import logging
import client
from faker import Faker
import pandas as pd
import data.plot as dpl


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
    def sessions(self):
        return SessionsGroup(
            [Session(steps) for steps in self.metrics_by_user_id.values()]
        )

    def plot_metrics(self, item_ids=[], figsize=(20, 6)):
        plot_ndcg_sessions(
            {
                n_steps: sessions.steps_mean_ndcg
                for n_steps, sessions in self.sessions.group_by_steps_count.items()
            },
            smooth_level=0.8,
            figsize=figsize,
        )

        plot_smooth_line(
            self.sessions.steps_by_index.mean_ndcg,
            xlabel="Session step",
            ylabel="Mean user NDGC",
            title="Mean User NDGC by Session Step",
            smooth_level=1,
            figsize=figsize,
        )

        plot_smooth_line(
            self.sessions.steps_by_index.mean_average_precision,
            xlabel="Session step",
            ylabel="Mean Average Precision",
            title="Mean User Average Precision by Session Step",
            smooth_level=1,
            figsize=figsize,
        )

        plot_smooth_line(
            self.sessions.steps_by_index.mean_reciprocal_rank,
            xlabel="Session step",
            ylabel="Mean Reciprocal Rank",
            title="Mean User Reciprocal Rank by Session Step",
            smooth_level=1,
            figsize=figsize,
        )

        plot_smooth_line(
            self.sessions.steps_by_index.mean_recall,
            xlabel="Session step",
            ylabel="Mean Recall",
            title="Mean User Recall by Session Step",
            smooth_level=1,
            figsize=figsize,
        )

        plot_n_users_by_session_evolution_size(
            [
                (n_steps, len(sessions))
                for n_steps, sessions in self.sessions.group_by_steps_count.items()
            ],
            figsize=figsize,
        )

        plt.show()

        dpl.describe_num_var_array(
            self.sessions.ndcg, "User Session Steps NDCG", figsize=figsize
        )

        dpl.describe_num_var_array(
            [len(s) for s in self.sessions],
            "User Session Steps Count",
            figsize=figsize,
        )

        dpl.describe_num_var_array(
            self.sessions.recall, "User Session Steps Recall", figsize=figsize
        )

        dpl.describe_num_var_array(
            self.sessions.mean_reciprocal_rank,
            "Mean Reciprocal Rank by USer Session",
            figsize=(20, 4),
        )

        dpl.describe_num_var_array(
            self.sessions.mean_average_precision,
            "Mean Average Precision by USer Session",
            figsize=(20, 4),
        )

        logging.info(f"User sessions Count: {len(self.sessions)}")
        logging.info(
            f"User sessions Steps Count: {len(self.sessions.steps)}"
        )

        logging.info(f"Mean NDCG: {self.sessions.mean_ndcg:.2}")
        logging.info(
            f"Mean Average Precision: {self.sessions.mean_mean_average_precision:.2}"
        )
        logging.info(
            f"Mean Reciprocal Rank: {self.sessions.mean_mean_reciprocal_rank:.2}"
        )
        logging.info(f"Mean Recall: {self.sessions.mean_recall:.2}")
        if len(item_ids) > 0:
            logging.info(
                f"Catalog Coverage: {self.sessions.catalog_coverage(item_ids):.2}"
            )
