import matplotlib.pyplot as plt
import logging
import pandas as pd
import data.plot as dpl
import util as ut


def build_path(base_path, filename, include_timestamp=True, format="%Y-%m-%d_%H-%M"):
    import matplotlib.pyplot as plt

    if base_path:
        path = ut.mkdir(f"{base_path}/")

        if include_timestamp:
            path += f"{ut.DateTimeUtils.now().strftime(format)}-"

        return path + filename.strip()
    else:
        return None

class SessionsPlotter:
    def __init__(self, sessions):
        self.sessions = sessions

    @property
    def count(self):
        return len(self.sessions)

    @property
    def total_steps_count(self):
        return len(self.sessions.steps)

    def plot_mean_ndcg_by_session_step_dist(
        self,
        figsize=(20, 6),
        save_path=None,
    ):
        dpl.describe_num_var_array(
            self.sessions.ndcg,
            "Mean NDCG by Session Step Distribution",
            figsize=figsize,
            output_path=build_path(save_path, "mean_ndcg_by_session_step_distribution"),
        )

    def plot_mean_ndcg_by_session_steps_series(
        self,
        figsize=(20, 6),
        min_sessions_by_step=4,
        save_path=None,
    ):
        dpl.plot_smooth_line(
            self.sessions.steps_by_index.mean_ndcg(
                min_sessions_by_step=min_sessions_by_step
            ),
            xlabel="Session Step",
            ylabel="Mean NDGC",
            title=f"Mean NDGC by Session Step (Min sessions by step = {min_sessions_by_step})",
            smooth_level=1,
            figsize=figsize,
            output_path=build_path(save_path, "mean_ndgc_by_session_step"),
        )

    def plot_mean_avg_precision_by_user_session_step_series(
        self,
        figsize=(20, 6),
        min_sessions_by_step=4,
        save_path=None,
    ):
        dpl.plot_smooth_line(
            self.sessions.steps_by_index.mean_average_precision(min_sessions_by_step),
            xlabel="Session Step",
            ylabel="Mean Average Precision",
            title=f"Mean Average Precision by Session Step Series (Min sessions by step = {min_sessions_by_step})",
            smooth_level=1,
            figsize=figsize,
            output_path=build_path(save_path, "mean_average_precision_by_session_step_series"),
        )


    def plot_mean_reciprocal_rank_by_user_session_step_series(
        self,
        figsize=(20, 6),
        min_sessions_by_step=4,
        save_path=None,
    ):
        dpl.plot_smooth_line(
            self.sessions.steps_by_index.mean_reciprocal_rank(min_sessions_by_step),
            xlabel="Session Step",
            ylabel="Mean Reciprocal Rank",
            title=f"Mean Reciprocal Rank by Session Step Series (Min sessions by step = {min_sessions_by_step})",
            smooth_level=1,
            figsize=figsize,
            output_path=build_path(save_path, "mean_reciprocal_rank_by_session_step_series"),
        )


    def plot_mean_recall_by_user_session_step(
        self,
        figsize=(20, 6),
        min_sessions_by_step=4,
        save_path=None,
    ):
        dpl.plot_smooth_line(
            self.sessions.steps_by_index.mean_recall(min_sessions_by_step),
            xlabel="Session Step",
            ylabel="Mean Recall",
            title=f"Mean Recall by Session Step Series (Min sessions by step = {min_sessions_by_step})",
            smooth_level=1,
            figsize=figsize,
            output_path=build_path(save_path, "mean_recall_by_session_step_series"),
        )


    def metrics(self, item_ids=[]):
        logging.info(f"Sessions: {self.count}")
        logging.info(
            f"Max Session Steps Reached: {len(self.sessions.steps_by_index.items)}"
        )

        logging.info(
            f"Sessions count >= 30 rated items: {self.sessions.n_session_with_more_than(n_items=30)}"
        )

        logging.info(f"Total Session Steps: {self.total_steps_count}")

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

    def plot_n_steps_by_session_dist(
        self,
        figsize=(20, 6),
        save_path=None,
    ):
        dpl.describe_num_var_array(
            [len(s) for s in self.sessions],
            "Steps Count by Session Distribution",
            figsize=figsize,
            output_path=build_path(save_path, "steps_count_by_session_distribution"),
        )

    def plot_n_found_relevant_items_by_session_step_dist(
        self,
        figsize=(20, 6),
        save_path=None,
    ):
        dpl.describe_num_var_array(
            self.sessions.n_found_relevant_items,
            "Found items Count by Session Step Distribution",
            figsize=figsize,
            output_path=build_path(save_path, "found_items_count_by_session_step_distribution"),
        )

    def plot_n_rated_items_by_session_dist(
        self,
        figsize=(20, 6),
        min_found_relevant_items=0,
        save_path=None,
    ):
        result = [
            len(s.found_relevant_items)
            for s in self.sessions
            if len(s.found_relevant_items) >= min_found_relevant_items
        ]

        if len(result) > 0:
            dpl.describe_num_var_array(
                result,
                f"Rated items by Session Distribution (min rated items: {min_found_relevant_items})",
                figsize=figsize,
                output_path=build_path(save_path, "rated_items_by_session_distribution"),
            )

    def bar_plot_sessions_by_step(
        self,
        figsize=(20, 6),
        save_path=None,
    ):
        dpl.dict_barplot(
            [
                (n_steps, len(steps))
                for n_steps, steps in self.sessions.steps_by_index.items
            ],
            xlabel="Session Step",
            ylabel="Sessions Count",
            title="Sessions Count by Session Step",
            figsize=figsize,
            output_path=build_path(save_path, "sessions_count_by_session_step"),
        )

    def plot_user_sessions_mean_average_precision(
        self,
        figsize=(20, 6),
        save_path=None,
    ):
        dpl.describe_num_var_array(
            self.sessions.mean_average_precision,
            "Mean Average Precision by Session Distribution",
            figsize=figsize,
            output_path=build_path(save_path, "mean_average_precision_by_session_distribution"),
        )


    def plot_user_sessions_mean_reciprocal_rank(
        self,
        figsize=(20, 6),
        save_path=None,
    ):
        dpl.describe_num_var_array(
            self.sessions.mean_reciprocal_rank,
            "Mean Reciprocal Rank by Session Distribution",
            figsize=figsize,
            output_path=build_path(save_path, "mean_reciprocal_rank_by_session_distribution"),
        )

    def plot_user_session_steps_recall_dist(
        self,
        figsize=(20, 6),
        save_path=None,
    ):
        dpl.describe_num_var_array(
            self.sessions.recall,
            "Mean Recall by Session Step Distribution",
            figsize=figsize,
            output_path=build_path(save_path, "mean_recall_by_session_step_distribution"),
        )

    def plot_n_found_relevant_items_segments_by_step(
        self,
        figsize=(20, 6),
        save_path=None,
    ):
        df = self.sessions.n_found_relevant_items_by_step_index.groupby(
            ["step_index", "n_found_relevant_items"], as_index=False
        ).size()

        dpl.stacked_barplot(
            df,
            x="step_index",
            hue="n_found_relevant_items",
            y="size",
            title="Relevant items found by step index for all sessions",
            xlabel="Session Step",
            ylabel="Found Relevant Items Count",
            legend="Found Relevant Items Count Segment",
            figsize=figsize,
            output_path=build_path(save_path, "relevant_items_found_by_step_index_for_all_sessions"),
        )

    def plot(
        self,
        item_ids=[],
        min_sessions_by_step=4,
        min_sessions=2,
        figsize=(20, 6),
        save_path=None,
    ):
        if len(self.sessions) == 0:
            logging.info("Not found sessions")
            return None

        self.metrics(item_ids)

        self.plot_n_steps_by_session_dist(
            figsize=figsize,
            save_path=save_path,
        )

        self.bar_plot_sessions_by_step(
            figsize=figsize,
            save_path=save_path,
        )

        self.plot_n_rated_items_by_session_dist(
            figsize=figsize,
            save_path=save_path,
        )

        self.plot_n_rated_items_by_session_dist(
            figsize=figsize,
            min_found_relevant_items=30,
            save_path=save_path,
        )

        self.plot_n_found_relevant_items_by_session_step_dist(
            figsize=figsize,
            save_path=save_path,
        )

        self.plot_n_found_relevant_items_segments_by_step(
            figsize=(figsize[0] * 0.825, figsize[1] * 1.3),
            save_path=save_path,
        )

        self.plot_mean_ndcg_by_session_step_dist(
            figsize=figsize,
            save_path=save_path,
        )

        self.plot_mean_ndcg_by_session_steps_series(
            figsize=figsize,
            min_sessions_by_step=min_sessions_by_step,
            save_path=save_path,
        )

        self.plot_user_sessions_mean_average_precision(
            figsize=figsize,
            save_path=save_path,
        )

        self.plot_mean_avg_precision_by_user_session_step_series(
            figsize=figsize,
            min_sessions_by_step=min_sessions_by_step,
            save_path=save_path,
        )

        self.plot_user_sessions_mean_reciprocal_rank(
            figsize=figsize,
            save_path=save_path,
        )

        self.plot_mean_reciprocal_rank_by_user_session_step_series(
            figsize=figsize,
            min_sessions_by_step=min_sessions_by_step,
            save_path=save_path,
        )

        self.plot_user_session_steps_recall_dist(
            figsize=figsize,
            save_path=save_path,
        )

        self.plot_mean_recall_by_user_session_step(
            figsize=figsize,
            save_path=save_path,
        )

        plt.show()
