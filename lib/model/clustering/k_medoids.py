import seaborn as sns
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import silhouette_visualizer


class KMedoisResult:
    def __init__(self, model, X, labels):
        self.model     = model
        self.X         = X
        self.labels    = labels

    def plot_clusters(self, dpi = 100):
        plt.rcParams['figure.dpi'] = dpi
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels)
        plt.title('Clusters')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show(block=False)
        return self


    def plot_silhouette(self):
        silhouette_visualizer(
            self.model,
            self.X,
            colors='yellowbrick'
        )
        return self

    def plot(self):
        self.plot_clusters()
        self.plot_silhouette()
        print(f'Silhouette: {silhouette_score(self.X, self.labels, metric="cosine")}')
        return self


class KMedoisClustering:
    def __init__(
        self,
        n_clusters   = 3,
        init         = 'k-medoids++'
    ):
        self.model =  KMedoids(
            n_clusters   = n_clusters,
            init         = init
        )

    def predict(self, X, verbose=False):
        labels = self.model.fit_predict(X)
        return KMedoisResult(self.model, X, labels)