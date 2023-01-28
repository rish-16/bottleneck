from enum import Enum, auto

from tasks.dictionary_lookup import DictionaryLookupDataset

import torch
from torch import nn
import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden, classes, heads=4):
        super().__init__()
        self.gnn = tgnn.Sequential('x, edge_index', [
                        (GCNConv(in_dim, hidden), 'x, edge_index -> x'),
                        nn.ReLU(inplace=True),
                        (GCNConv(hidden, hidden), 'x, edge_index -> x'),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden, classes),
                    ])

        self.tf = nn.MultiheadAttention(in_dim, heads, batch_first=True) # 147, 128
        self.project = nn.Linear(in_dim, classes)

    def forward(self, x, edge_index):
        x_gnn = self.gnn(x, edge_index)
        x_tf, weights = self.tf(x, x, x)
        x_proj = self.project(x_tf)

        comb = x_gnn + x_proj
        out = torch.relu(comb)

        return out

class Task(Enum):
    NEIGHBORS_MATCH = auto()

    @staticmethod
    def from_string(s):
        try:
            return Task[s]
        except KeyError:
            raise ValueError()

    def get_dataset(self, depth, train_fraction):
        if self is Task.NEIGHBORS_MATCH:
            dataset = DictionaryLookupDataset(depth)
        else:
            dataset = None

        return dataset.generate_data(train_fraction)


class GNN_TYPE(Enum):
    GCN = auto()
    GGNN = auto()
    GIN = auto()
    GAT = auto()
    GT = auto()

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, in_dim, out_dim):
        if self is GNN_TYPE.GCN:
            return GCNConv(
                in_channels=in_dim,
                out_channels=out_dim)
        elif self is GNN_TYPE.GGNN:
            return GatedGraphConv(out_channels=out_dim, num_layers=1)
        elif self is GNN_TYPE.GIN:
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                         nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()))
        elif self is GNN_TYPE.GAT:
            # 4-heads, although the paper by Velickovic et al. had used 6-8 heads.
            # The output will be the concatenation of the heads, yielding a vector of size out_dim
            num_heads = 4
            return GATConv(in_dim, out_dim // num_heads, heads=num_heads)
        elif self is GNN_TYPE.GT:
            return GraphTransformer(in_dim, hidden=32, heads=4, classes=out_dim)

class STOP(Enum):
    TRAIN = auto()
    TEST = auto()

    @staticmethod
    def from_string(s):
        try:
            return STOP[s]
        except KeyError:
            raise ValueError()

def one_hot(key, depth):
    return [1 if i == key else 0 for i in range(depth)]
