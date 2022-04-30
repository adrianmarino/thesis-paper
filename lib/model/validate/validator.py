import data as dt
import data.dataset as ds
from .validator_summary import ValidatorSummary


class Validator:
    def __init__(self, n_samples, batch_size, metrics_fn, predictors):
        self.n_samples  = n_samples
        self.batch_size = batch_size
        self.metrics_fn = metrics_fn
        self.predictors = predictors

    def validate(self, ds, **kwargs):
        summary = []
        with dt.progress_bar(self.n_samples) as bar:
            for sample in range(self.n_samples):
                X, y_true = ds.sample(self.batch_size)

                for p in self.predictors:
                    y_pred = p.predict_batch(X, **kwargs)

                    metrics = {name: fn(y_pred, y_true) for name, fn in self.metrics_fn.items()}
                    metrics['predictor'] = p.name
                    metrics['sample']    = sample

                    summary.append(metrics)
                bar()

        return ValidatorSummary(summary)