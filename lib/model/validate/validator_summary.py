import pandas as pd
import data.plot as pl
import data as dt
import numpy as np

class ValidatorSummary:

    @staticmethod
    def load(file_path): return ValidatorSummary(pd.read_json(f'{file_path}.json'))

    @staticmethod
    def from_list(metrics_log): return ValidatorSummary(pd.DataFrame(metrics_log))

    def save(self, file_path): self.data.to_json(f'{file_path}.json')

    @classmethod
    def load_and_join(clazz, paths):
        return clazz.join([clazz.load(path) for path in paths])

    @staticmethod
    def join(summaries):
        return ValidatorSummary(pd.concat([s.data for s in summaries], ignore_index=True))

    def __init__(self, data): self.data = data

    def __predictor_names(self):
        return np.unique(self.data['predictor'].values)

    def __metric_names(self):
        return set(self.data.columns) - set(['predictor', 'sample'])

    def plot(
        self,
        bins                  = 10,
        show_table            = False,
        show_range            = False,
        include_found_metrics = False
    ):
        metric_names = self.__metric_names()
        if not include_found_metrics:
            metric_names = [m for m in metric_names if not 'found' in m]

        predictor_names = self.__predictor_names()

        for metric in metric_names:
            pl.xl_flat_size()
            pl.comparative_boxplot(self.data, x=metric, y="predictor", title=f'Metric: {metric}')

        for metric_name in metric_names:
            for pre_name in predictor_names:
                pl.xl_flat_size()
                pl.describe_num_var(
                    df         = self.data[self.data.predictor == pre_name],
                    column     = metric_name,
                    bins       = bins,
                    title      = f'Model: {pre_name} - Metric: {metric_name}',
                    show_table = show_table,
                    show_range = show_range
                )

    def to_latex(self):
        return self.show().style.format(thousands = ".").to_latex()

    def show(
        self,
        sort_columns          = ['mAP@5(4,5)'],
        select_columns        = [],
        include_found_metrics = False,
        ascending             = False
    ):
        columns = self.data.columns

        if select_columns:
            columns = ['predictor'] + select_columns
        elif not include_found_metrics:
            columns = [c for c in columns if not 'found' in c]

        print(f'Ordered by {", ".join(sort_columns)}:')
        return self \
                .data[columns] \
                .groupby('predictor') \
                .mean() \
                .sort_values(by=sort_columns, ascending=ascending)