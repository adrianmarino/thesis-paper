import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def lineplot(
    data,
    x,
    y,
    x_rotation     = 60,
    figsize        = (15, 6),
    title          = '',
    title_fontsize = 20,
    axis_fontsize  = 16
):
    sns.lineplot(x=data[x],  y=data[y].astype('float'))

    sns.set(rc={'figure.figsize': figsize})
    plt.xticks(rotation=x_rotation)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(x, fontsize=axis_fontsize)
    plt.ylabel(y, fontsize=axis_fontsize)
    plt.show(block=False)


def moving_avg(df, column, window): df[column] = df[column].rolling(window=window).mean()

def plot_lines(
    lines,
    hue,
    xlabel            = 'x',
    ylabel            = 'y',
    title             = '',
    moving_avg_window = 15,
    fill              = False,
    alpha             = 1.0,
    area_alpha        = 0.5
):

    count = min([len(line) for line in lines.values()])

    data = {label: line[:count] for label, line in lines.items() }

    data['x'] = list(range(1, count+1))

    df = pd.DataFrame(data)

    [moving_avg(df, column, moving_avg_window) for column in lines.keys()]

    df = pd.melt(
        df, 
        id_vars     = ['x'], 
        value_vars  = lines.keys(), 
        var_name    = hue, 
        value_name  = 'y'
    )
    
    sns.lineplot(
        data  = df,
        x     = 'x',
        y     = 'y',
        hue   = hue,
        alpha = alpha
    )
    
    if fill:
        for column in lines.keys():
            line_df = df[df[hue] == column]
            
            plt.fill_between(
                line_df['x'].values, 
                line_df['y'].values, 
                alpha=area_alpha,
            )
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)