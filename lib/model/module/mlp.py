from torch.nn import Module, Linear, BatchNorm1d, ReLU, Dropout, Sequential
from pytorch_common.modules import CommonMixin



class MultiLayerPerceptron(Module, CommonMixin):
    def __init__(
        self,
        units_per_layer : list[int]   = [10, 2],
        activation      : list        = [ReLU(), ReLU()],
        dropout         : list[float] = [0.2],
        batch_norm      : list[bool]  = [True]
    ):
        super().__init__()

        layers = []
        for i in range(len(units_per_layer)-1):
            n_input, n_output = units_per_layer[i:i+2]

            layers.append(Linear(n_input, n_output))

            if i < len(units_per_layer)-2:
                if batch_norm and i < len(batch_norm) and batch_norm[i]:
                    layers.append(BatchNorm1d(n_output))

                if activation and i < len(activation) and activation[i]:
                    layers.append(activation[i])

                if dropout and i < len(dropout) and dropout[i]:
                    layers.append(Dropout(p=dropout[i]))

            elif activation and i < len(activation) and activation[i]:
                    layers.append(activation[i])

        self.mlp = Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
