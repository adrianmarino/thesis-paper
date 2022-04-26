import pandas as pd
from .rules import ColumnMissingRules, ColumnMissingRule
from data import dtype
import logging


def is_empty(df):
    return df.shape[0] == 1 and df.shape[1] == 1


class MissingsManager:
    def __init__(self, rules = ColumnMissingRules()):
        self._rules   = rules

    def add_rule(self, column, rule): 
        self._rules.add_rule(column, rule)
        return self
    
    def add_rule_fn(self, column, fn): 
        self._rules.add_rule_fn(column, fn)
        return self

    def report(self, df, verbose=1):
        data = []
        for column in df.columns:
            col_type   = dtype(df[column])
            rule       = self._rules.by(column, col_type)
            col_values = df[column].values
            
            indexes  = []
            for index, col_value in enumerate(col_values):
                if rule.is_missing(col_value):
                    indexes.append(index)
        
            total          = len(col_values)
            missings_count = len(indexes)

            if missings_count > 0:
                row = {'Column': column, 'Percent (%)': missings_count / total, 'Count': missings_count}
                if verbose:
                    row['rule'] = rule.__class__.__name__
                    row['indexes'] = indexes
                data.append(row)
        
        return pd.DataFrame(data if len(data) > 0 else [{'Column': 'Not found columns with missing values'}])


    def remove_rows(self, df, max_missings=0.05):
        """
        Remove dataset rows with missings values associated to columns with <= max_missings % of missing values".
        """
        report = self.report(df, verbose=1)
        if is_empty(report): 
            return df

        logging.info(f'Remove rows for columns <= {max_missings} % of missings values.')
        indexes_lists = report[report['Percent (%)'] <= max_missings]['indexes']

        indexes = set([idx for idxs in indexes_lists for idx in idxs])

        return df.drop(indexes, axis=0)

    def remove_columns(self, df, max_percent=0.4):
        report = self.report(df, verbose=1)
        if is_empty(report): 
            return df

        logging.info(f'Remove columns with missing >= {max_percent} %')
        
        columns = report[report['Percent (%)'] >= max_percent]['Column']

        return df.drop(columns, axis=1)
