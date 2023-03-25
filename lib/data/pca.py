from sklearn import decomposition
import pandas as pd

import numpy as np
import pandas as pd

import data.plot as pl
import util as ut

import seaborn as sns
import matplotlib.pyplot as plt


class PCASummary:
    def __init__(self, pc, pca, y, n_components, columns):
        self.pc_columns          = [f'pc{i}'for i in range(1, n_components+1)]
        self.__observations      = pd.DataFrame(data = pc, columns = self.pc_columns)
        self.__observations['y'] = y
        self.pca                 = pca
        self.__columns           = columns

    def explained_variance(self):
        return self.pca.explained_variance_ratio_


    def plot_explained_variance(self):
        pl.barplot(
            data       = pd.DataFrame({
                'Variance': self.pca.explained_variance_ratio_,
                'Principal Components': self.pc_columns
            }),
            x          = 'Principal Components',
            y          = 'Variance',
            x_rotation = 0
        )
        return self

    def plot_clusters(
        self,
        pc_x       = 'pc1',
        pc_y       = 'pc2',
        fit_reg    = False,
        legend     = True,
        point_size = 40
    ):
        sns.lmplot(
            x           = pc_x,
            y           = pc_y,
            data        = self.__observations,
            fit_reg     = fit_reg,
            hue         = 'y',
            legend      = legend,
            scatter_kws = { "s": point_size}
        )
        plt.show()
        return self

    def components(self, indexes=[0, 1]):
        return np.transpose(self.pca.components_[indexes, :])

    def observations(self, indexes=[1, 2]):
        return self.__observations[[f'pc{i}' for i in indexes]].values

    def biplot(
        self,
        comps      = [1, 2],
        labels     = None,
        s          = 300,
        alpha      = 0.2,
        edgecolors = 'none'
    ):
        plt.style.use('ggplot')
        score  = self.observations(indexes=comps)
        coeff  = self.components()
        y      = self.__observations['y'].values
        labels = self.__columns

        xs = score[:,0]
        ys = score[:,1]

        n  = coeff.shape[0]

        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())

        fig, ax = plt.subplots()
        ax.scatter(
            xs * scalex,
            ys * scaley,
            c          = pd.factorize(y)[0],
            s          = s,
            alpha      = alpha,
            edgecolors = edgecolors
        )
        ax.legend()
        ax.grid(True)

        for i in range(n):
            ax.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r', alpha = 0.8)
            if labels is None:
                ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))

        plt.show()
        return self

    def plot(self):
        return self.plot_clusters() \
        .biplot() \
        .plot_explained_variance() \
        .explained_variance()


class PCAAnalisys:
    @staticmethod
    def make_on(X: pd.DataFrame, y: np.array, n_components=4):
        pca = decomposition.PCA(n_components=n_components)

        num_X = X.select_dtypes(include=np.number)
        pc = pca.fit_transform(num_X.values)

        return PCASummary(pc, pca, y, n_components, X.columns)