from torch.nn import  Linear, BatchNorm1d, ReLU, Dropout, Sigmoid


class LayerGroup:
    @staticmethod
    def linearBatchNormReluDropout(self, input_size, output_size, dropout=0.2):
        return [
            Linear(input_size, output_size),
            BatchNorm1d(output_size),
            ReLU(True),
            Dropout(dropout)
        ]

    @staticmethod
    def linearBatchNormSigmoid(self, input_size, output_size):
        return [
            Linear(input_size, output_size),
            BatchNorm1d(output_size),
            Sigmoid()
        ]