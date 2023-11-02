import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE


class TSNEAnalisys:
    @staticmethod
    def make_on(
        X: pd.DataFrame,
        y: np.array,
        n_components = 3,
        random_state = 42,
        n_iter       = 1000,
        n_jobs       = 24
    ):
        model = TSNE(
            n_components = n_components,
            n_iter       = n_iter,
            n_jobs       = n_jobs,
            random_state = random_state
        )

        result = model.fit_transform(X)

        columns=[f'C{idx}' for idx in range(1, n_components+1)]
        result_df = pd.DataFrame(result, columns=columns)
        result_df['class'] = y

        return TSNEAnalisysSymmary(result_df, model.kl_divergence_)


class TSNEAnalisysSymmary:
    def __init__(self, data, kl_divergence):
        self.data = data
        self.kl_divergence = kl_divergence

    def plot(
        self,
        width        = 800,
        height       = 800,
        opacity      = 0.7,
        marker_size  = 0.1
    ):
        fig = px.scatter_3d(
            self.data,
            x        = 'C1',
            y        = 'C2',
            z        = 'C3',
            color    = 'class',
            width    = width,
            height   = height,
            opacity  = opacity
        )

        fig.update_traces(marker_size = 12)

        fig.update_layout(
            title=f"t-SNE clusters visualization (kl_divergence_={self.kl_divergence})"
        )

        fig.show()