import numpy as np
import pandas as pd

MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']
WEEK   = ['mon', 'tue', 'wed', 'thus', 'fri', 'sat', 'sun']

is_list = lambda type_: type_ != object and type_ != str and type_ == list


def dtype(series):
    if series.dtype == object and len(series.dropna()) > 0:
        value = series.dropna()[0]
        if type(value) == np.ndarray:
            return list

        if type(value) != str and type(value) == list:
            return list

    return series.dtype


def is_nan_array(value):
    if type(value) == list or type(value) == np.ndarray:
        return len(value) == 0
    else:
        response = np.isnan(value)

        if type(np.isnan(response)) == np.bool_:
            return response
        elif type(np.isnan(response)) == np.ndarray:
            return len(response) == 0


def frequency(array, name,  ascending=False):
    unique, counts = np.unique(array, return_counts=True)
    df = pd.DataFrame(
        np.asarray((unique, counts)).T, 
        columns=[name, 'count']
    )
    df['count'] = df['count'].astype('long')  
    return df.sort_values(by=['count'], ascending=ascending)


def group_by(df, column, asc_order=False):
    return df \
    .groupby(column) \
    .size() \
    .reset_index(name='count') \
    .sort_values(by='count', ascending=asc_order)


def list_column_to_dummy_columns(df, column, prefix=None):
    data =  df.drop(column, 1).join(df[column].str.join('|').str.get_dummies())

    new_columns = list(set(data.columns) - set(df.columns))
    
    if prefix:
        data = data.rename(columns={c: f'{prefix}_{c}' for c in new_columns})

    return data