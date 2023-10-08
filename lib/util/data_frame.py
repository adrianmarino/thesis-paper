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



def get_dummies_from_list_col(df, source, prefix=''):
    target_cols = np.unique(np.concatenate(df[source].apply(np.array).values))

    target_col_values = {c:[] for c in target_cols}

    for _, row in df.iterrows():
        values = row[source]

        for col in values:
            target_col_values[col].append(1)

        for col in set(target_cols) - set(values):
            target_col_values[col].append(0)

    result = df.copy()
    for col in target_col_values.keys():
        result[f'{prefix}_{col.lower()}' if prefix  else col.lower()] = target_col_values[col]

    return result



def embedding_from_list_col(df, id_col, value_col, exclude=[]):
    df = df[[id_col, value_col]]
    df = get_dummies_from_list_col(df, value_col)
    df = df.drop(columns=[value_col]+exclude)
    df = df.groupby([id_col]).sum().reset_index()

    emb_columns = sorted(list(set(df.columns) - set([id_col])))

    emb_df = df[emb_columns]
    emb_df = emb_df.apply(lambda row: row / row.sum(), axis=1)
    emb_df = emb_df.dropna()

    result_df = pd.DataFrame()
    result_df[f'{value_col}_embedding'] = emb_df[emb_columns].apply(lambda row: row.tolist(), axis=1)
    result_df.insert(0, id_col, df[id_col])

    result_df

    return result_df


def year_to_decade(df, source, target):
    df[target] = df[source].apply(lambda year: int(year / 10) * 10)
    return df


def group_mean(df, group_col, mean_col):
    return df.groupby([group_col])[mean_col].mean().reset_index()


def mean_by_key(df, key, value):
    ut.to_dict(group_mean(df, key, value), key, value)