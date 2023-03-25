import random
import numpy as np
import torch


def set_seed(value = 42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)