import matplotlib.pyplot as plt
from joypy import joyplot
from matplotlib import cm


def ridgeplot(
    df,
    by,
    column,
    title='',
    kind = 'kde'
):
    joyplot(
        df,
        by          = by,
        column      = column,
        colormap    = cm.autumn,
        fade        = True,
        range_style = 'own',
        title       = title,
        kind        = kind
    )
    plt.show(block=False)
