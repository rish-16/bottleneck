import torch
from torch import nn
from torchinfo import summary
import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden, heads, classes):
        super().__init__()
        self.gnn = tgnn.Sequential('x, edge_index', [
                        (GCNConv(in_dim, hidden), 'x, edge_index -> x'),
                        nn.ReLU(inplace=True),
                        (GCNConv(hidden, hidden), 'x, edge_index -> x'),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden, classes),
                    ])

        self.tf = nn.MultiheadAttention(in_dim, heads) # 147, 128
        self.project = nn.Linear(in_dim, classes)

    def forward(self, x, edge_index):
        x_gnn = self.gnn(x, edge_index)
        x_tf, weights = self.tf(x, x, x)
        x_proj = self.project(x_tf)

        comb = x_gnn + x_proj
        out = torch.relu(comb)

        return out

x = torch.rand(147, 128)
edge_index = torch.randint(0, 147, [2, 70])
edge_index = tg.utils.to_undirected(edge_index)
gt = GraphTransformer(128, 64, 4, classes=8)
y = gt(x, edge_index)

print (summary(gt))
print (x.shape, edge_index.shape)
print (y.shape)