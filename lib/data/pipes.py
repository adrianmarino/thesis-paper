import data as dt
import numpy as np
import pandas as pd
import torch


select   = lambda df, columns: df[columns]
distinct = lambda df, columns: df.drop_duplicates(subset=columns)
rename   = lambda df, mapping: df.rename(columns=mapping)
drop     = lambda df, columns: df.drop(columns=columns, errors='ignore')


def tf_idf(df, column, tfIdf_generator = dt.TfIdfGenerator()):
    return tfIdf_generator(df[column].values)


def reset_index(df):
    df.reset_index(drop=True, inplace=True)
    return df


def tokenize(df, column, tokenizer = dt.TokenizerService()):
    df[f'{column}_tokens'] = df[column].apply(lambda x: ' '.join(tokenizer(x)))
    return df


def join_str_list(df, column, join_str=' '):
    df[column] = df[column].apply(lambda it: join_str.join(it))
    return df


def append_emb_vectors(df, embedding, column):
    df['temp_seq'] = df.index
    df[f'{column}_embedding'] = df['temp_seq'].apply(lambda it: embedding[it, :].cpu().detach().numpy() if torch.is_tensor(embedding[it, :]) else embedding[it, :])
    return drop(df, ['temp_seq'])


def sum_cols(df, sources, target='sum', dtype=np.float64):
    total = pd.Series(dtype=dtype)
    for col in sources:
        total += df[col]
    df[target] = total
    return df

def concat_columns(df, column_a, column_b, out_column=None, separator=' '):
    if out_column == None:
        out_column = column_a
    df[out_column] = df[column_b].astype(str) + separator + df[column_a].astype(str)
    return df
