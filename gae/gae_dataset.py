import torch
from torch.utils.data import Dataset
import numpy as np


class GAEDataset(Dataset):
    def __init__(self, adj, adj_label, features):
        assert adj.shape[0] == adj.shape[1] and adj.shape[0] == adj_label.shape[0] and adj.shape[0] == features.shape[0]
        self.adj = adj
        self.adj_label = adj_label
        self.features = features
        self.inds = torch.from_numpy(np.array(range(features.shape[0]), dtype=np.float32))

    def __getitem__(self, index):
        adj_row = self.adj[index]
        adj_label_row = self.adj_label[index]
        feature = self.features[index]
        ind = self.inds[index]

        return adj_row, adj_label_row, feature, ind

    def __len__(self):
        return self.features.shape[0]