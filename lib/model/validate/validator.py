import data as dt
import data.dataset as ds
from .validator_summary import ValidatorSummary
import torch


class Validator:
    def __init__(self, n_samples, batch_size, metrics, predictors):
        self.n_samples  = n_samples
        self.batch_size = batch_size
        self.metrics    = metrics
        self.predictors = predictors

    def validate(self, ds,  **kwargs):
        summary = []
        with dt.progress_bar(self.n_samples) as bar:
            for sample in range(self.n_samples):
                X, y_true = ds.sample(self.batch_size)

                for p in self.predictors:
                    y_pred = p.predict_batch(X, **kwargs)

                    metrics = {m.name: m.perform(y_pred, y_true, X).item() for m in self.metrics}
                    metrics['predictor'] = p.name
                    metrics['sample']    = sample

                    summary.append(metrics)
                bar.update()

        return ValidatorSummary(summary)