import pandas as pd
import torch
import data.dataset as ds


def to_tensor(obs, device, columns): 
    data = obs[columns]
    if type(data) == pd.DataFrame:
        data = data.values
    return torch.tensor(data).int().to(device)


class DatasetFactory:
    def __init__(
        self,
        feature_columns = ['user_seq', 'item_seq'],
        target_column   = 'rating'
    ): 
        self.__features_fn = lambda obs, device: to_tensor(obs, device, feature_columns)
        self.__target_fn   = lambda obs, device: to_tensor(obs, device, [target_column])   

    def create_from(self, df: pd.DataFrame):
        return ds.RecSysDataset(df, self.__features_fn, self.__target_fn)