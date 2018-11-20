"""Class for loading NMT dataset."""

import torch
import numpy as np
from torch.utils.data import Dataset


class NMTDataset(Dataset):

    """Class to load NMT dataset."""

    def __init__(self, X, y, X_id2token, y_id2token, X_token2id, y_token2id,
                 max_sent_len):
        """
        Initialize NMTDataset.

        Args
            X: list of index sequences from the first language.
            y: list of index sequences from the second language.
        """
        self.X = {i: sent for i, sent in enumerate(X)}
        self.y = {i: sent for i, sent in enumerate(y)}
        self.X_id2token = X_id2token
        self.y_id2token = y_id2token
        self.X_token2id = X_token2id
        self.y_token2id = y_token2id
        self.max_sent_len = max_sent_len

    def randomize_samples(self, k=10):
        """Return index batches of inputs."""
        indexes = [x for x in range(len(self))]
        np.random.shuffle(indexes)
        s = 0
        size = int(np.ceil(len(indexes) / k))
        batches = []
        while s < len(indexes):
            batches += [indexes[s:s + size]]
            s = s + size
        return batches

    def __len__(self):
        """Return length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        X = self.X[i]
        y = self.y[i]

        X_len = min(len(X), self.max_sent_len)
        y_len = min(len(y), self.max_sent_len)

        X_pad = np.zeros(self.max_sent_len)
        for j in range(X_len):
            X_pad[j] = X[j]

        y_pad = np.zeros(self.max_sent_len)
        for j in range(y_len):
            y_pad[j] = y[j]

        X_pad = torch.from_numpy(X_pad).long()
        y_pad = torch.from_numpy(y_pad).long()
        X_len = torch.from_numpy(np.array(X_len)).long()
        y_len = torch.from_numpy(np.array(y_len)).long()

        return {'X': X_pad, 'y': y_pad, 'X_len': X_len, 'y_len': y_len}
