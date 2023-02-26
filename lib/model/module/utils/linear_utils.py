class LinearUtils:
    @staticmethod
    def init_weights(linear, initrange=0.1):
        linear.bias.data.zero_()
        linear.weight.data.uniform_(-initrange, initrange)
        return linear