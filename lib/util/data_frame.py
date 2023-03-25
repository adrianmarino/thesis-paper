import pandas as pd


norm = lambda df: (df - df.mean()) / df.std()


def id_by_seq(df, entity=None, column_id=None, column_seq=None):
    if entity:
        column_id  = f'{entity}_id'
        column_seq = f'{entity}_seq'
    return to_dict(df, key=column_seq, value=column_id)


def to_dict(df, key, value):
    #                  entry.value                  entry.key
    return pd.Series(df[value].values, index=df[key]).to_dict()