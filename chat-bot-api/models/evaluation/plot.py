import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pytorch_common.util as pu
import util as ut
import numpy as np


def smooth_lineplot(
    x,
    y,
    smooth_level              = 0.7,
    confidence_interval_alpha = 0.15,
    label                     = ''
):
    y_smooth = gaussian_filter1d(
        y,
        sigma = smooth_level
    )

    error = np.abs(y_smooth - y)

    sns.lineplot(
        x     = x,
        y     = y_smooth,
        label = label
    )
    plt.fill_between(
        x,
        y_smooth - error,
        y_smooth + error,
        alpha = confidence_interval_alpha
    )


def plot_smooth_line(
    values,
    xlabel,
    ylabel,
    title,
    smooth_level = 2,
    figsize      = (5,3)
):
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)

    smooth_lineplot(
        x                         = [k for k, _ in values.items()],
        y                         = [v for _, v in values.items()],
        smooth_level              = smooth_level
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def plot_ndcg_sessions(
    ndcgs_by_sessions_size,
    smooth_level = 0.8,
    figsize      =(14, 5)
):
    plt.figure(figsize=figsize)

    for size, ndcgs in sorted(ndcgs_by_sessions_size.items()):
        smooth_lineplot(
            x                         = list(range(1, len(ndcgs)+1)),
            y                         = ndcgs,
            label                     = f'{size} Session steps)',
            smooth_level              = smooth_level
        )

    plt.xlabel('User Session step')
    plt.ylabel('NDGC')
    plt.title('NDGC by User Session Step')
    plt.legend()


def plot_n_users_by_session_evolution_size(
    users_by_sessions_size,
    figsize =(5 ,4)
):
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x = [i[0] for i in users_by_sessions_size],
        y = [i[1] for i in users_by_sessions_size],
        estimator="sum"
    )
    ax.bar_label(ax.containers[0], fontsize=10);
    plt.xlabel("Sessions steps")
    plt.ylabel("Users Count")
    plt.title("Users Count by Session Steps")