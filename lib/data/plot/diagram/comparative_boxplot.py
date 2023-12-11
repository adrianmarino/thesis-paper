import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def comparative_boxplot(
    data,
    x,
    y,
    title          = '',
    title_fontsize = 20,
    axis_fontsize  = 16,
    x_rotation     = 60,
    figsize        = (20, 6),
    output_path    = None,
    output_ext     = 'svg',
    ascending      = False
):
    mean_by_x = data.groupby([y])[x].median().sort_values(ascending=ascending)

    sns.boxplot(x=x,  y=y, data=data,  order=mean_by_x.index)

    plt.xticks(rotation=x_rotation)
    sns.set(rc={'figure.figsize': figsize})
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(x, fontsize=axis_fontsize)
    plt.ylabel(y, fontsize=axis_fontsize)
    if output_path:
        plt.savefig(f'{output_path}.{output_ext}', format=output_ext)
    plt.show(block=False)
