import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_common.util as pu
import util as ut
import numpy as np
import pandas as pd
import math


def barplot(
    data,
    x,
    y,
    xlabel=None,
    ylabel=None,
    title="",
    title_fontsize=20,
    axis_fontsize=16,
    x_rotation=60,
    figsize=(15, 6),
    output_path=None,
    output_ext="svg",
    instant_plot=True,
):
    sns.barplot(x=data[x], y=data[y].astype("float"))
    plt.xticks(rotation=x_rotation)
    sns.set(rc={"figure.figsize": figsize})
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(x if xlabel is None else xlabel, fontsize=axis_fontsize)
    plt.ylabel(y if ylabel is None else ylabel, fontsize=axis_fontsize)

    if output_path:
        plt.savefig(f"{output_path}.{output_ext}", format=output_ext)


    if instant_plot:
        plt.show(block=False)


def dict_barplot(
    values_by_index,
    xlabel="",
    ylabel="",
    title="",
    title_fontsize=16,
    axis_fontsize=16,
    bar_fontsize=10,
    sort_by="key",
    ascending=True,
    figsize=(5, 4),
    output_path=None,
    output_ext="svg",
    instant_plot=False,
):
    df = pd.DataFrame(values_by_index, columns=["key", "value"])
    df = df.sort_values(by=sort_by, ascending=ascending)

    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df, x="key", y="value", order=df[sort_by])

    ax.bar_label(ax.containers[0], fontsize=bar_fontsize)
    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.ylabel(ylabel, fontsize=axis_fontsize)
    plt.title(title, fontsize=title_fontsize)

    if output_path:
        plt.savefig(f"{output_path}.{output_ext}", format=output_ext)

    if instant_plot:
        plt.show(block=False)


def stacked_barplot(
    df,
    x,
    y,
    hue,
    title="",
    title_fontsize=16,
    axis_fontsize=16,
    legend_fontsize=12,
    bar_fontsize=7,
    legend="",
    xlabel=None,
    ylabel=None,
    figsize=(10, 6),
    output_path=None,
    output_ext="svg",
    instant_plot=True,
):
    df_pivot = df.pivot(index=x, columns=hue, values=y)

    ax = df_pivot.plot(kind="bar", stacked=True, figsize=figsize)

    bottom = np.zeros(len(df_pivot))
    for col in df_pivot.columns:
        for idx, value in enumerate(df_pivot[col]):
            if not math.isnan(value) and not math.isnan(bottom[idx]):
                ax.text(
                    x=idx,
                    y=bottom[idx] + value / 2,
                    s=str(round(value)),
                    ha="center",
                    va="center",
                )

        bottom += df_pivot[col].values

    plt.legend(title=legend, fontsize=legend_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.ylabel(ylabel, fontsize=axis_fontsize)

    plt.tight_layout()

    if output_path:
        plt.savefig(f"{output_path}.{output_ext}", format=output_ext)

    if instant_plot:
        plt.show()
