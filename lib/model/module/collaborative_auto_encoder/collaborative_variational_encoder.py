
import torch
from torch.nn import Module, ReLU, Embedding, Linear, Dropout
from pytorch_common.modules import CommonMixin, PredictMixin
import logging
from ..distribution import DistributionFactory


class CollaborativeVariationalEncoder(Module, CommonMixin, PredictMixin):
    def __init__(
        self,
        n_item_ratings,
        dropout                 = 0.2,
        activation              = None,
        mu_simgma_dim     : int = 512,
        latent_space_dim  : int = 256
    ):
        super().__init__()
        self.type = 'CollaborativeVariationalEncoder'

        self.lineal       = Linear(n_item_ratings, mu_simgma_dim)
        self.dropout      = Dropout(dropout) if dropout else None
        self.activation   = activation

        self.mu_linea     = Linear(mu_simgma_dim, latent_space_dim)
        self.sigma_lineal = Linear(mu_simgma_dim, latent_space_dim)
        self.normal       = DistributionFactory.normal()
        self.kl           = 0


    def forward(self, input_data, verbose=False):
        """
        input  = [batch_size, n_item_ratings]
        output = [batch_size, latent_space_dim]
        """
        input_data = input_data.to(self.device)

        # Get inputs...
        if verbose:
            logging.info(f'{self.type} - Input: {input_data.shape}')

        x     = self.lineal(input_data)
        if self.dropout:
            x = self.dropout(x)
        x     = self.activation(x)

        # Tenemos un valor de medio(mu) y varianza(sigma) para cada dimensión del espacio latente.
        mu    = self.mu_linea(x)
        sigma = torch.exp(self.sigma_lineal(x))

        if verbose:
            logging.info(f'{self.type} - mu: {mu.shape}')
            logging.info(f'{self.type} - sigma: {sigma.shape}')

        # Sampling de valores aleatorios de la dict N(mu, sigma) a partir de valrres de una dist N(0, 1).
        # En este caso sampleamos mu.shape valores, uno para cada dimension de mu y sigma.
        # Tengamos en cuenta que mu y sigma xy z vectores de tamaño mu.shape. Cada dimension es una variable
        # aleatoria que sigue una distribucion normal con sus propios parametro mu y sigma.
        # El modelo se encarga aprender los parametro mu y sigmas para cada dimension del espacio latente.
        z = mu + sigma*self.normal.sample(mu.shape)

        if verbose:
            logging.info(f'{self.type} - z: {z.shape}')

        # Luego calculamos la divergencia de Kullback-Leibler (KL) para usarla en la funcion de loss
        # como un termino mas. A menor divergencia, mas se asemejara la distribucion D(mu, sigma)
        # a una distribucion normal estadar N(0, 1). Esto permite que:
        #
        # - Se conserven las distancias de los puntos del espacio de entrada en el espacio latente,
        #   preservando así la estructura de los datos en un espaciode menor dimensionalidad.
        # - Obliga al modelo a producir distribuciones latentes que se parezcan a una distribución
        #   normal estándar. Esto evita que el modelo aprenda a codificar información irrelevante o ruidosa
        #   en la representación latente. De esta manera, se puede obtener una representación latente
        #   más robusta y generalizable que pueda ser utilizada para generar nuevas muestras similares
        #   a las originales.
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        self.kl = (torch.pow(sigma,2) + torch.pow(mu, 2) - torch.log(sigma) - 1/2).sum()

        if verbose:
            logging.info(f'{self.type} - KL: {self.kl}')

        return z