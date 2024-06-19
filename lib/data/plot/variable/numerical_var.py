from ..diagram import histplot
import pandas as pd


def array_num_to_str(numbers):
    return [str(n) for n in numbers]


def describe_num_var_array(
    values,
    title='',
    figsize=None,
    title_fontsize=16,
):
    describe_num_var(
        pd.DataFrame({title: values}),
        column=title,
        title=f"{title} (observations: {len(values)})",
        title_fontsize=title_fontsize,
        figsize=figsize,
    )


def describe_num_var(
    df,
    column,
    bins="auto",
    stat="count",
    title="",
    title_fontsize=16,
    show_table=False,
    show_range=False,
    show_mean=True,
    show_median=True,
    show_mode=True,
    show_outliers_leyend=True,
    remove_outliers=False,
    decimals=3,
    density=True,
    output_path=None,
    output_ext="svg",
    figsize=None,
):
    if show_range or show_table:
        df_column = df[[column]]

    if show_range:
        column_range = (min(df_column.values)[0], max(df_column.values)[0])
        print(f"\nRange: {column_range}\n")

    histplot(
        df,
        column,
        bins,
        stat,
        title,
        title_fontsize,
        show_mean,
        show_median,
        show_mode,
        show_outliers_leyend,
        remove_outliers,
        decimals,
        density=density,
        output_path=output_path,
        output_ext=output_ext,
        figsize=figsize,
    )

    if show_table:
        print("\nMetrics:\n")
        display(df_column.describe())
