import pandas as pd


norm = lambda df: (df - df.mean()) / df.std()


def df_to_dict(df, key, value):
    return pd.Series(df[value].values,index=df[key]).to_dict()