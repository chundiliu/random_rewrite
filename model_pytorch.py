import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN_reimp(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim):
        super(GCN_reimp, self).__init__()
        self.fc1 = nn.Linear(input_feat_dim, hidden_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        hidden_emb = F.normalize(x, p=2, dim=1)
        adj_preds = F.relu(torch.mm(hidden_emb, hidden_emb.t()))
        return  adj_preds, hidden_emb


