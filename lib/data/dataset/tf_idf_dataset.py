from torch.utils.data import Dataset
import torch

class TfIdfDataset(Dataset):
    def __init__(self, matrix): 
        self.data = torch.tensor(matrix.toarray(), dtype=torch.float)

    def __len__(self): return self.data.shape[0]

    @property
    def shape(self): return self.data.shape

    def __getitem__(self, idx): return self._row(idx), self._row(idx)

    def _row(self, idx): return self.data[idx, :].squeeze(0)