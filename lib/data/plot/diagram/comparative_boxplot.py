import seaborn as sns
import matplotlib.pyplot as plt


def comparative_boxplot(
    data,
    x,
    y,
    title          = '',
    title_fontsize = 20,
    axis_fontsize  = 16,
    x_rotation     = 60,
    figsize        = (15, 6),
    output_path    = None,
    output_ext     = 'svg'
):
    sns.boxplot(x=x,  y=y, data=data)

    plt.xticks(rotation=x_rotation)
    sns.set(rc={'figure.figsize': figsize})
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(x, fontsize=axis_fontsize)
    plt.ylabel(y, fontsize=axis_fontsize)
    if output_path:
        plt.savefig(f'{output_path}.{output_ext}', format=output_ext)
    plt.show(block=False)
