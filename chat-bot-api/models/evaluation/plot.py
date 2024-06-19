import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pytorch_common.util as pu
import util as ut
import numpy as np
import pandas as pd
import util as ut


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

    plt.xlabel('Session step')
    plt.ylabel('Mean NDGC')
    plt.title('Mean NDGC by Session Step')
    plt.legend()



def bar_plot(
    values_by_index,
    xlabel='',
    ylabel='',
    title='',
    sort_by='value',
    ascending=True,
    estimator="sum",
    figsize =(5 ,4)
):
    df = pd.DataFrame(values_by_index, columns=['key', 'value'])
    df = df.sort_values(by=sort_by, ascending=ascending)

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x = 'key',
        y = 'value',
        order= df[sort_by],
        estimator=estimator
    )
    ax.bar_label(ax.containers[0], fontsize=10);
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def bar_plot_df(
    df,
    x,
    y,
    sort_by,
    xlabel='',
    ylabel='',
    title='',
    estimator="sum",
    figsize =(5 ,4)
):
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x = x,
        y = y,
        order= df[sort_by],
        estimator=estimator
    )
    ax.bar_label(ax.containers[0], fontsize=10);
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)