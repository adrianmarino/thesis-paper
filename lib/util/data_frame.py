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



def embedding_from_list_col(df, id_col, value_col, exclude=[], as_list=True):
    # for each id_col, compute total count for each value into value_col...
    df = df[[id_col, value_col]]
    df = get_dummies_from_list_col(df, value_col)
    df = df.drop(columns=[value_col]+exclude, errors='ignore')
    df = df.groupby([id_col]).sum().reset_index()

    # Remove id_col column...
    emb_columns = sorted(list(set(df.columns) - set([id_col])))

    # Normalize each row by total roe count...
    emb_df = df[emb_columns]
    emb_df = emb_df.apply(lambda row: row / row.sum(), axis=1)
    emb_df = emb_df.dropna()

    # Add id_col column...
    result_df = pd.DataFrame()

    if as_list:
        result_df[f'{value_col}_embedding'] = emb_df[emb_columns].apply(lambda row: row.tolist(), axis=1)
    else:
        result_df = emb_df

    result_df.insert(0, id_col, df[id_col])

    return result_df


def get_one_hot_from_list_col(df, id_col, value_col, exclude=[], as_list=True):
    # for each id_col, compute total count for each value into value_col...
    df = df[[id_col, value_col]]
    df = get_dummies_from_list_col(df, value_col)
    df = df.drop(columns=[value_col]+exclude, errors='ignore')

    # Add id_col column...
    result_df = pd.DataFrame()

    if as_list:
        # Remove id_col column...
        emb_columns = sorted(list(set(df.columns) - set([value_col, id_col])))

        result_df[f'{value_col}_one_hot'] = df[emb_columns].apply(lambda row: row.tolist(), axis=1)
        result_df.insert(0, id_col, df[id_col])
    else:
        result_df = df

    result_df = result_df.drop_duplicates(subset=[id_col])

    return result_df


def year_to_decade(df, source, target):
    df[target] = df[source].apply(lambda year: int(year / 10) * 10)
    return df


def group_mean(df, group_col, mean_col):
    return df.groupby([group_col])[mean_col].mean().reset_index()


def mean_by_key(df, key, value):
    ut.to_dict(group_mean(df, key, value), key, value)


def column_types(df):
    types_list = df  \
        .dtypes \
        .to_frame(name='type') \
        .reset_index() \
        .rename(columns= {'index': 'column'}) \
        .to_dict('records')

    result = {}

    for row in types_list:
        if row['type'] == np.dtypes.ObjectDType:
            v = next(v for v in df[row['column']].values if v is not None)
            if hasattr(v, '__len__'):
                result[row['column']] = 'list'
        elif type(row['type']) == np.dtypes.DateTime64DType:
            result[row['column']] = 'datetime'
        elif type(row['type']) == np.dtypes.BoolDType:
            result[row['column']] = 'bool'
        elif type(row['type']) == np.dtypes.Int64DType:
            result[row['column']] = 'int'
        elif type(row['type']) == np.dtypes.Float64DType:
            result[row['column']] = 'float'
        elif row['type'] == 'string':
            result[row['column']] = 'str'
        else:
            result[row['column']] = row['type']

    return result



def one_hot(df, cols, col_bucket={}):
    col_types = column_types(df)

    result = df

    for col in cols:
        if col in col_types:
            col_type = col_types[col]

            if col_type == 'list':
                result = get_dummies_from_list_col(result, col, col)

            elif col_type in ['str', 'bool']:
                result = pd.get_dummies(result, columns=[col], prefix=col)

            elif col_type == 'int':
                if col in col_bucket:
                    bucket_size = col_bucket[col]
                    result[f'{col}_bucket'] = result[col].apply(lambda v: int(v / bucket_size) * bucket_size)

                result = pd.get_dummies(result, columns=[f'{col}_bucket'], prefix=col)

    return result

def multiply_by(df, columns, by_column):
    return df[columns].multiply(df[by_column], axis="index")


def group_sum(df, group_col):
    return df.groupby([group_col]).sum().reset_index()



def bins_column(df, column, bins):
    bins = np.concatenate([[0], bins, [np.inf]])

    def skip_empty_desimals(value):
        return str(value).rstrip('0').rstrip('.')

    prev_bin = bins[0]
    labels   = []
    for curr_bin in bins[1:]:
        if curr_bin == np.inf:
            labels.append(f'{skip_empty_desimals(prev_bin)}+')
        else:
            labels.append(f'{skip_empty_desimals(prev_bin)}-{skip_empty_desimals(curr_bin-1)}')
            prev_bin = curr_bin

    df[f'{column}_bin'] = pd.cut(
        df[column],
        bins           = bins,
        labels         = labels,
        include_lowest = True
    )

    return df