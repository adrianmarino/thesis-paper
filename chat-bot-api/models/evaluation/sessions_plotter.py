from .plot import (
    smooth_lineplot,
    plot_smooth_line,
    plot_ndcg_sessions,
    plot_n_users_by_session_evolution_size,
)
import matplotlib.pyplot as plt
import logging
import pandas as pd
import data.plot as dpl


class SessionsPlotter:
    def __init__(self, sessions):
        self.sessions = sessions

    @property
    def count(self):
        return len(self.sessions)

    @property
    def total_steps_count(self):
        return len(self.sessions.steps)

    def plot_mean_ndcg_evolution(
        self,
        figsize=(20, 6),
    ):
        plot_smooth_line(
            self.sessions.steps_by_index.mean_ndcg,
            xlabel="User Session Step",
            ylabel="NDGC",
            title="NDGC by User Session Step",
            smooth_level=1,
            figsize=figsize,
        )

    def plot_mean_avg_precision_evolution(
        self,
        figsize=(20, 6),
    ):
        plot_smooth_line(
            self.sessions.steps_by_index.mean_average_precision,
            xlabel="User Session Step",
            ylabel="Mean Average Precision",
            title="Mean Average Precision by User Session Step",
            smooth_level=1,
            figsize=figsize,
        )

    def plot_mean_reciprocal_rank_evolution(
        self,
        figsize=(20, 6),
    ):
        plot_smooth_line(
            self.sessions.steps_by_index.mean_reciprocal_rank,
            xlabel="User Session Step",
            ylabel="Mean Reciprocal Rank",
            title="Mean User Reciprocal Rank by User Session Step",
            smooth_level=1,
            figsize=figsize,
        )

    def plot_mean_recall_evolution(
        self,
        figsize=(20, 6),
    ):
        plot_smooth_line(
            self.sessions.steps_by_index.mean_recall,
            xlabel="User Session Step",
            ylabel="Mean Recall",
            title="Mean User Recall by User Session Step",
            smooth_level=1,
            figsize=figsize,
        )

    def metrics(self, item_ids=[]):
        logging.info(f"User Sessions - Count: {self.count}")
        logging.info(f"User Sessions - Steps Count: {self.total_steps_count}")

        logging.info(f"User Sessions - NDCG: {self.sessions.mean_ndcg:.2}")
        logging.info(
            f"User Sessions - Mean Average Precision: {self.sessions.mean_mean_average_precision:.2}"
        )
        logging.info(
            f"User Sessions - Mean Reciprocal Rank: {self.sessions.mean_mean_reciprocal_rank:.2}"
        )
        logging.info(f"User Sessions - Recall: {self.sessions.mean_recall:.2}")
        if len(item_ids) > 0:
            logging.info(
                f"Catalog Coverage: {self.sessions.catalog_coverage(item_ids):.2}"
            )

    def plot(self, item_ids=[], figsize=(20, 6)):
        self.metrics(item_ids)

        dpl.describe_num_var_array(
            [len(s) for s in self.sessions],
            "User Session Steps Count",
            figsize=figsize,
        )

        plot_n_users_by_session_evolution_size(
            [
                (n_steps, len(sessions))
                for n_steps, sessions in self.sessions.group_by_steps_count.items()
            ],
            figsize=figsize,
        )

        dpl.describe_num_var_array(
            self.sessions.ndcg, "User Session Steps NDCG", figsize=figsize
        )

        self.plot_mean_ndcg_evolution(figsize=figsize)

        plot_ndcg_sessions(
            {
                n_steps: sessions.steps_mean_ndcg
                for n_steps, sessions in self.sessions.group_by_steps_count.items()
            },
            smooth_level=0.8,
            figsize=figsize,
        )

        dpl.describe_num_var_array(
            self.sessions.mean_average_precision,
            "User Sessions Mean Average Precision",
            figsize=figsize,
        )

        self.plot_mean_avg_precision_evolution(figsize=figsize)

        dpl.describe_num_var_array(
            self.sessions.mean_reciprocal_rank,
            "User Sessions Mean Reciprocal Rank",
            figsize=figsize,
        )

        self.plot_mean_reciprocal_rank_evolution(figsize=figsize)

        dpl.describe_num_var_array(
            self.sessions.recall, "User Session Steps Recall", figsize=figsize
        )

        self.plot_mean_recall_evolution(figsize=figsize)

        plt.show()
