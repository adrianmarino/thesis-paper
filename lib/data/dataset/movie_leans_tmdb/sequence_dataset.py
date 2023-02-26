from torch.utils.data import Dataset
import data as dt
import util as ut
import torch


class SequenceDataset(Dataset):
    @staticmethod
    def load(path):
        ut.mkdir(path)
        return SequenceDataset(
            ut.PickleUtils.load(f'{path}/features'),
            ut.PickleUtils.load(f'{path}/targets'),
            ut.PickleUtils.load(f'{path}/id_by_seq'),
            ut.PickleUtils.load(f'{path}/seq_by_id')
        )

    def __init__(self, features, targets, id_by_seq, seq_by_id):
        self.features  = features
        # self.targets   = (targets < 1000).int()
        self.targets   = targets
        self.id_by_seq = id_by_seq
        self.seq_by_id = seq_by_id

    def __len__(self):
        return self.targets.shape[0]

    @property
    def shape(self):
        return (self.targets.shape, self.targets.shape)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def save(self, path):
        ut.mkdir(path)
        ut.PickleUtils.save(self.features,  f'{path}/features')
        ut.PickleUtils.save(self.targets,   f'{path}/targets')
        ut.PickleUtils.save(self.id_by_seq, f'{path}/id_by_seq')
        ut.PickleUtils.save(self.seq_by_id, f'{path}/seq_by_id')
