import numpy as np


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
