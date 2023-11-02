from bunch import Bunch
import numpy as np
from sklearn.metrics import silhouette_score
import data as dt
import matplotlib.pyplot   as plt
import seaborn             as sns


class KSilhouette:
    def __init__(self, k_by_silhouette):
        self.ks              = list(k_by_silhouette.values())
        self.silhouettes     = list(k_by_silhouette.keys())

    def better_k(self):
        return self.ks[np.argmax(self.silhouettes)]

    def plot(self):
        ay = sns.lineplot(x= self.ks, y=self.silhouettes, markers=True, dashes=False)
        ay.set(title='Silhouette by clusters count (More is better)')
        ay.set(xticks= self.ks)
        plt.show()


class SilhouetteAnalisys:
    @staticmethod
    def make_on(
        X,
        train_model_fn,
        max_clusters=10
    ):
        k_values     = list(range(2, max_clusters+1))
        k_by_silhouette = {}

        with dt.progress_bar(len(k_values), 'Compute silhouette') as bar:
            for k in k_values:
                model = train_model_fn(X, k)

                silhouette = silhouette_score(X, model.labels_)

                k_by_silhouette[silhouette] = k
                bar.update()

        return KSilhouette(k_by_silhouette)