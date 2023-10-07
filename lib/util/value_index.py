import pandas as pd


class ValueIndex:
    def __init__(
        self, 
        df, 
        value_col,
        index_cols
    ):  
        index = pd.MultiIndex.from_arrays(
            [df[col].values for col in index_cols],
            names=index_cols
        )
        self.df = pd.DataFrame({value_col: df[value_col].values}, index=index)
        self.value_col = value_col

    def __getitem__(self, index_tuples):
        return self.df.loc[index_tuples][self.value_col].values

    def __repr__(self): return str(self.df)
    def __str__(self): return  str(self.df)