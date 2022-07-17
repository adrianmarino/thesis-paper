import matplotlib.pyplot as plt
from joypy import joyplot


def ridgeplot(
    df,
    by,
    column,
    title=''
):
    joyplot(
        df,
        by          = by,
        column      = column,
        colormap    = cm.autumn,
        fade        = True,
        range_style = 'own',
        title       = title)
    plt.show()
