import data as dt
import data.dataset as ds
from .validator_summary import ValidatorSummary
import torch
import torch.multiprocessing as mp
import os


def evaluate_fn(p, y_true, X, sample, metrics, kwargs):
    y_pred = p.predict_batch(X, **kwargs)

    result = { 'predictor': p.name, 'sample': sample}
    for m in metrics:
        result.update(m.perform(y_pred, y_true, X))

    return result


class Validator:
    def __init__(self, n_samples, batch_size, metrics, predictors, n_processes=5, target_transform_fn = lambda it: it.squeeze(1)):
        self.n_samples   = n_samples
        self.batch_size  = batch_size
        self.metrics     = metrics
        self.predictors  = predictors
        self.n_processes = n_processes
        self.target_transform_fn = target_transform_fn

    def validate(self, ds,  **kwargs):
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        summary = []
        with dt.progress_bar(self.n_samples) as bar:
            for sample in range(self.n_samples):
                X, y_true = ds.sample(self.batch_size)
                y_true    = self.target_transform_fn(y_true)

                for p in self.predictors:
                    summary.append(evaluate_fn(p, y_true, X, sample, self.metrics, kwargs))

                bar.update()


        return ValidatorSummary.from_list(summary)