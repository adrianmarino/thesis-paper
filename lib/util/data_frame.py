import pandas as pd
import numpy as np


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



def get_dummies_from_list_col(df, column, prefix=''):
    columns = np.unique(np.concatenate(df[column].apply(np.array).values))

    column_values = {c:[] for c in columns}

    for _, row in df.iterrows():
        values = row[column]

        for v in values:
            column_values[v].append(1)

        for v in set(columns) - set(values):
            column_values[v].append(0)

    result = df.copy()
    for name in column_values.keys():
        col = f'{prefix}_{name.lower()}' if prefix  else name.lower() 
        result[col] = column_values[name]

    return result



def embedding_from_list_col(df, id_col, value_col, exclude=[]):
    df = df[[id_col, value_col]]
    df = get_dummies_from_list_col(df, value_col)
    df = df.drop(columns=[value_col]+exclude)
    df = df.groupby([id_col]).sum().reset_index()

    emb_columns = sorted(list(set(df.columns) - set([id_col])))

    emb_df = df[emb_columns]
    emb_df = emb_df.apply(lambda row: row / row.sum(), axis=1)

    result_df = pd.DataFrame()
    result_df[f'{value_col}_embedding'] = emb_df[emb_columns].apply(lambda row: row.tolist(), axis=1)
    result_df.insert(0, id_col, df[id_col])

    return result_df
