import torch
from torch.utils.data import Dataset
import pytorch_common.util as pu


class BasicDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, feat_type=torch.float, target_type=torch.float):
        self.features = torch.tensor(df[feature_cols].to_numpy()).type(feat_type)
        self.target   = torch.tensor(df[target_col].to_numpy()).type(target_type)

    def __len__(self): return self.target.shape[0]

    @property
    def shape(self): return self.target.shape

    def __getitem__(self, idx): return self.features[idx], self.target[idx]

    def sample(self, size):
        indexes = torch.randint(0, len(self)-1, (size,))
        return self.features[indexes], self.target[indexes]
