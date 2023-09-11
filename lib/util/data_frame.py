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


def save_df(df, path): return df.to_json  (path, orient='records')
def load_df(path):     return pd.read_json(path, orient='records')


def datetime_to_seq(df, source, target):
    # Convert str source column to datetime
    datetime_col = pd.to_datetime(df[source])

    target_df = df.copy()

    # Add unique seq number using rank()
    target_df[target]=  datetime_col.rank(method='dense').astype(int)

    return target_df