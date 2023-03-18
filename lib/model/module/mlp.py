from torch.nn import Module, Linear, BatchNorm1d, ReLU, Dropout, Sequential
from pytorch_common.modules import CommonMixin


class MultiLayerPerceptron(Module, CommonMixin):
    def __init__(
        self,
        input_units     : int,
        units_per_layer : list[int],
        dropout         : float = None,
        batch_norm      : bool  = True,
        n_outputs       : int=1
    ):
        super().__init__()
        layers = []

        assert len(units_per_layer) > 1, "MPL must have more than one layer!"

        for units in units_per_layer:
            layers = [Linear(input_units, units)]

            if batch_norm:
                layers.append(BatchNorm1d(units))

            layers.append(ReLU())

            if dropout:
                layers.append(Dropout(p=dropout))

            last_units = units

        layers.append(Linear(last_units, n_outputs))

        self.mlp = Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
