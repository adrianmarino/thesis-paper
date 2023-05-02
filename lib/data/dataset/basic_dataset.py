import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, df, feature_columns, target_column):
        self.features = torch.tensor(df[feature_columns].to_numpy()).float()
        self.target   = torch.tensor(df[target_column].to_numpy()).float()

    def __len__(self): return self.target.shape[0]

    @property
    def shape(self): return self.target.shape

    def __getitem__(self, idx): return self.features[idx], self.target[idx]

    def sample(self, size):
        indexes = torch.randint(0, len(self)-1, (size,))
        return self.features[indexes], self.target[indexes]
