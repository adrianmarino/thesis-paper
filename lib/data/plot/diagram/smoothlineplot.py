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
    title_fontsize=16,
    axis_fontsize=16,
    smooth_level=0.7,
    confidence_interval_alpha=0.15,
    label="",
):
    y_smooth = gaussian_filter1d(y, sigma=smooth_level)

    error = np.abs(y_smooth - y)

    sns.lineplot(x=x, y=y_smooth, label=label)
    plt.fill_between(
        x, y_smooth - error, y_smooth + error, alpha=confidence_interval_alpha
    )


def plot_smooth_line(
    values,
    xlabel,
    ylabel,
    title,
    legend="",
    title_fontsize=16,
    axis_fontsize=16,
    smooth_level=2,
    figsize=(5, 3),
    output_path=None,
    output_ext="svg",
):
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)

    smooth_lineplot(
        x=[k for k, _ in values.items()],
        y=[v for _, v in values.items()],
        smooth_level=smooth_level,
    )

    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.ylabel(ylabel, fontsize=axis_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(legend, fontsize=axis_fontsize)

    # Ajustar los márgenes
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)


    if output_path:
        plt.savefig(f'{output_path}.{output_ext}', format=output_ext)