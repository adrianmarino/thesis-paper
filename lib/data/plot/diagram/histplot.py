import seaborn as sns
import matplotlib.pyplot as plt
from data import outliers_range, mode
import numpy as np


def histplot(
    df,
    column,
    bins                 = 'auto',
    stat                 = 'count',
    title                = '',
    title_fontsize       = 16,
    show_mean            = True,
    show_median          = True,
    show_mode            = True,
    show_outliers_leyend = True,
    remove_outliers      = False,
    decimals             = 10,
    density              = True,
    output_path          = None,
    output_ext           = 'svg',
    figsize              = None,
    instant_plot         = False,
    axis_fontsize        = 16
):
    f, (ax_box, ax_hist) = plt.subplots(
        2,
        sharex=True,
        gridspec_kw= {"height_ratios": (0.2, 1)}
    )

    if figsize:
        f.set_size_inches(figsize[0], figsize[1])

    values = df[column].values
    outyliers_lower, outliers_upper = outliers_range(values)

    if remove_outliers:
        values = df[(df[column]>=outyliers_lower) & (df[column]<=outliers_upper)][column].values

    mean       = np.mean(values)
    median     = np.median(values)
    mode_value = mode(values)

    sns.boxplot(x=values, ax=ax_box)
    if show_mean:
        ax_box.axvline(mean,   color='r', linestyle='--')
    if show_median:
        ax_box.axvline(median, color='g', linestyle='-')
    if show_mode:
        ax_box.axvline(mode_value,  color='b', linestyle='-')
    ax_box.set_title(f'Boxplot')
    ax_box.set(xlabel='')


    sns.histplot(x=values, ax=ax_hist, bins=bins, kde=density)


    if show_mean:
        ax_hist.axvline(mean,   color='r', linestyle='--', label=f'Mean ({round(mean, decimals)})')
    if show_median:
        ax_hist.axvline(median, color='g', linestyle='-',  label=f'Median ({round(median, decimals)})')
    if show_mode:
        ax_hist.axvline(mode_value,   color='b', linestyle='-',  label=f'Mode ({round(mode_value, decimals)})')
    if show_outliers_leyend and not remove_outliers:
        outyliers_lower_percent = (len([v for v in values if v <= outyliers_lower])/len(values))*100
        outliers_upper_percent = (len([v for v in values if v >= outliers_upper])/len(values))*100

        ax_hist.axvline(outyliers_lower,  color='black', linestyle='-', label=f'Outliers lower ({round(outyliers_lower, decimals)} - {outyliers_lower_percent:.2f}%)')
        ax_hist.axvline(outliers_upper,   color='black', linestyle='-', label=f'Outliers Upper ({round(outliers_upper, decimals)} - {outliers_upper_percent:.2f}%)')


    ax_hist.legend(fontsize=axis_fontsize)

    ax_hist.set_title(f'Histogram')
    ax_hist.set_ylabel('Frequency', fontsize=axis_fontsize)
    ax_hist.set_xlabel(column, fontsize=axis_fontsize)

    title = title if title else column
    if remove_outliers:
        title  += ' (Without Outliers)'

    f.suptitle(title, fontsize=title_fontsize)

    # Ajustar los márgenes
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)

    if output_path:
        plt.savefig(f'{output_path}.{output_ext}', format=output_ext)

    if instant_plot:
        plt.show(block=False)
