import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from joypy import joyplot
from matplotlib import cm


def headmap(matrix, title='', figsize=(10, 10), hide_labels=True, title_fontsize=20):
    sns.heatmap(matrix, xticklabels=not hide_labels, yticklabels=not hide_labels)

    sns.set(rc={'figure.figsize': figsize})
    plt.title(title, fontsize=title_fontsize)
    plt.show()


def barplot(data, x, y, x_rotation=60, figsize=(15, 6), title='', title_fontsize=20, axis_fontsize=16):
    sns.barplot(x=data[x],  y=data[y].astype('float'))

    plt.xticks(rotation=x_rotation)
    sns.set(rc={'figure.figsize': figsize})
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(x, fontsize=axis_fontsize)
    plt.ylabel(y, fontsize=axis_fontsize)
    plt.show()

    
def lineplot(data, x, y, x_rotation=60, figsize=(15, 6), title='', title_fontsize=20, axis_fontsize=16):
    sns.lineplot(x=data[x],  y=data[y].astype('float'))

    sns.set(rc={'figure.figsize': figsize})
    plt.xticks(rotation=x_rotation)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(x, fontsize=axis_fontsize)
    plt.ylabel(y, fontsize=axis_fontsize)
    plt.show()

    
def histplot(df, column, bins='auto', stat='count'):
    f, (ax_box, ax_hist) = plt.subplots(
        2, 
        sharex=True, 
        gridspec_kw= {"height_ratios": (0.2, 1)}
    )
    mean=df[column].mean()
    median=df[column].median()
    mode=df[column].mode().values[0]

    sns.boxplot(data=df, x=column, ax=ax_box)
    ax_box.axvline(mean,   color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')
    ax_box.axvline(mode,   color='b', linestyle='-')

    sns.histplot(data=df, x=column, ax=ax_hist, kde=True, bins=bins, stat=stat)
    ax_hist.axvline(mean,   color='r', linestyle='--', label="Mean")
    ax_hist.axvline(median, color='g', linestyle='-',  label="Median")
    ax_hist.axvline(mode,   color='b', linestyle='-',  label="Mode")

    ax_hist.legend()

    ax_hist.set_title(f'Histogram - {column}')
    ax_hist.set(ylabel='Frequency')
    ax_hist.set(xlabel='Values')

    ax_box.set_title(f'Boxplot - {column}')
    ax_box.set(xlabel='')

    plt.show()


def words_clous_plot(
    words,
    min_font_size    = 7, 
    random_state     = 21, 
    max_font_size    = 50, 
    relative_scaling = 0.5, 
    colormap         = 'Dark2',
    title            = '', 
    title_fontsize   = 20
):
    if type(words) != list:
        raise Exception("Words arg must be a strings list!")

    if len(words) == 0: 
        return

    if type(words[0]) != str:
        raise Exception("Words arg must be a strings list!")
   
    text = " ".join(words)
    
    word_cloud = WordCloud(
        min_font_size    = min_font_size, 
        random_state     = random_state, 
        max_font_size    = max_font_size, 
        relative_scaling = relative_scaling, 
        colormap         = colormap
    ).generate(text)
    # Display the generated Word Cloud
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=title_fontsize)
    plt.show()


def ridgeplot(df, by, column, title=''):
    joyplot(
        df, 
        by          = by, 
        column      = column, 
        colormap    = cm.autumn, 
        fade        = True,
        range_style = 'own',
        title       = title)
    plt.show()