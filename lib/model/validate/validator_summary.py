import pandas as pd
import data.plot as pl
import data as dt
import numpy as np


class ValidatorSummary:
    def __init__(self, metrics_log):
        self.data = pd.DataFrame(metrics_log) 

    def plot(self, bins=10):
        metric_names = set(self.data.columns) - set(['predictor', 'sample'])   
        for name in metric_names:
            for pre_name in np.unique(self.data['predictor'].values):
                print(f'Predictor: {pre_name}')
                data = self.data[self.data.predictor == pre_name]

                pl.l_flat_size()
                pl.describe_num_var(data, name, bins=bins)
