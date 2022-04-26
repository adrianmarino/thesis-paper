from abc import ABC
import math
import pandas as pd
from data import is_nan_array, is_list


class ColumnMissingRule(ABC):
    def is_missing(self, value): pass


class ColumnMissingValueRule(ColumnMissingRule):
    def is_missing(self, value): return pd.isna(value)



class ColumnMissingListRule(ColumnMissingRule):
    def is_missing(self, value): return is_nan_array(value)


class CustomMissingFnRule(ColumnMissingRule):
    def __init__(self, fn):
        self._fn = fn
    
    
    def is_missing(self, value):
        return self._fn(value)

    
class ColumnMissingRules: 
    def __init__(
        self, 
        list_null  = ColumnMissingListRule(),
        value_null = ColumnMissingValueRule()
    ):
        self._rules      = {}
        self._list_null  = list_null
        self._value_null = value_null

    def add_rule(self, column, rule):
        self._rules[column] = rule

    def add_rule_fn(self, column, fn):
        self.add_rule(column, CustomMissingFnRule(fn))

    def by(self, column, dtype):
        if column not in self._rules:
            if is_list(dtype):
                self.add_rule(column, self._list_null)
            else:
                self.add_rule(column, self._value_null)

        return self._rules[column]

    def reset(self): self._rules.clear()