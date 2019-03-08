import torch
from torch.utils.data import Dataset
import numpy as np


class GAEDataset(Dataset):
    def __init__(self, adj, adj_label):
        assert adj.shape[0] == adj.shape[1] and adj.shape[0] == adj_label.shape[0]

        self.MAX_NNZ = max([rrow.getnnz() for rrow in adj])

        self.adj_inds = np.zeros(shape=[adj.shape[0], self.MAX_NNZ], dtype=np.int32)
        self.adj_vals = np.zeros(shape=[adj.shape[0], self.MAX_NNZ], dtype=np.float32)
        for i in range(adj.shape[0]):
            self.adj_inds[i, :] = adj[i].indices
            self.adj_vals[i, :] = adj[i].data

        self.adj_label_inds = np.zeros(shape=[adj_label.shape[0], self.MAX_NNZ], dtype=np.int32)
        self.adj_label_vals = np.zeros(shape=[adj_label.shape[0], self.MAX_NNZ], dtype=np.float32)
        for i in range(adj_label.shape[0]):
            self.adj_label_inds[i, :] = adj_label[i].indices
            self.adj_label_vals[i, :] = adj[i].data


        self.inds = torch.from_numpy(np.array(range(adj.shape[0]), dtype=np.float32))

    def __getitem__(self, index):
        adj_row = self.adj[index]
        adj_label_row = self.adj_label[index]
        feature = self.features[index]
        ind = self.inds[index]

        return adj_row, adj_label_row, feature, ind

    def __len__(self):
        return self.features.shape[0]