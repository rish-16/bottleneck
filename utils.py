import torch
from torch_scatter import scatter_add
import torch_geometric as tg

def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = torch.eye(n_nodes) - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD


def dirichlet_energy(X, edge_index):
    n_nodes = X.size(0)
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    adj = tg.utils.to_dense_adj(edge_index).squeeze(0).numpy().tolist()

    energy = 0

    for i in range(len(adj)):
        for j in range(len(adj[i])):
            if adj[i][j] == 1:
                wij = 1
                diff = ((X[i]/torch.sqrt(1 + deg[i])) - (X[j]/torch.sqrt(1 + deg[j])))
                energy += torch.pow(wij * torch.norm(diff, p=2), 2)

    energy = 0.5 * energy

    return energy

def jacobian():
    pass

from typing import Any

import math

import torch
from torch import Tensor


def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)


def kaiming_uniform(value: Any, fan: int, a: float):
    if isinstance(value, Tensor):
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            kaiming_uniform(v, fan, a)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            kaiming_uniform(v, fan, a)


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)


def zeros(value: Any):
    constant(value, 0.)


def ones(tensor: Any):
    constant(tensor, 1.)


def normal(value: Any, mean: float, std: float):
    if isinstance(value, Tensor):
        value.data.normal_(mean, std)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            normal(v, mean, std)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            normal(v, mean, std)


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)