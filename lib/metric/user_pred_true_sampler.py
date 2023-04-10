from random import sample
import logging
import torch
import numpy as np
import util as ut


class UserYPredYTrueSampler:
    """Sample a list of sample_size items for all user in ( y_true, y_pred) sets.
    Each sample is a subset of  (y_true, y_pred) for a given user.
    Create a sample for all users found in X dataset.
    """

    def __init__(self, user_index, sample_size, decimals=0):
        """Create a UserYPredYTrueSampler instance.

        Args:
            user_index (int): index of user ids into X matrix.
            sample_size (int): size of items sample.
            decimals (int, optional): Specify sample values decimals count. Defaults to 0 decimals.
        """
        self.__user_index = user_index
        self.__sample_size = sample_size
        self.__decimals = decimals

    def sample(self, y_true, y_pred, X):
        """For each user in X dataset get a sample from (y_true, y_pred) sets. 
        It sample is a subset of (y_true, y_pred) for a given user.

        Args:
            y_true (array): ground true values
            y_pred (array): predicted values.
            X (matrix): original dataset. User to get user ids / sequence numbers.

        Yields:
            (y_true, y_pred): a (y_true, y_pred) subser for a given user.
        """
        user_seq = X[:, self.__user_index]

        for user_idx in torch.unique(user_seq):
            y_true_sample, y_pred_sample = \
                self.__y_true_y_pred_sample_by_user_idx(y_true, y_pred, user_seq, user_idx)

            if y_true_sample == None:
                continue

            indexes_desc_by_val = torch.argsort(y_true_sample, dim=0, descending=True)

            yield y_true_sample[indexes_desc_by_val], y_pred_sample[indexes_desc_by_val]

    def __y_true_y_pred_sample_by_user_idx(self, y_true, y_pred, user_seq, user_idx):
        indexes = ut.indexes_of(user_seq, user_idx)

        if indexes.size()[0] < self.__sample_size:
            return None, None

        indexes_sample = ut.random_choice(indexes, self.__sample_size)

        y_true_sample = torch.index_select(y_true, dim=0, index=indexes_sample)
        if not ut.is_int(y_true_sample):
            y_true_sample = torch.round(y_true_sample, decimals=self.__decimals)

        y_pred_sample = torch.index_select(y_pred, dim=0, index=indexes_sample)
        if not ut.is_int(y_pred_sample):
            y_pred_sample = torch.round(y_pred_sample, decimals=self.__decimals)

        return y_true_sample, y_pred_sample
