from torch.nn import  Linear, BatchNorm1d, ReLU, Dropout, Sigmoid


class LayerGroup:
    @staticmethod
    def linearBatchNormReluDropout(input_size, output_size, dropout=0.2):
        return [
            Linear(in_features=input_size, out_features=output_size),
            BatchNorm1d(num_features=output_size),
            ReLU(True),
            Dropout(dropout)
        ]

    @staticmethod
    def linearBatchNormSigmoid(input_size, output_size):
        return [
            Linear(in_features=input_size, out_features=output_size),
            BatchNorm1d(num_features=output_size),
            Sigmoid()
        ]