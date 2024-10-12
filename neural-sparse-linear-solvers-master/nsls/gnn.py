from typing import Optional

from torch import Tensor
import torch.nn as nn
import torch_geometric.nn


class GraphBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.graph_norm = torch_geometric.nn.GraphNorm(width)
        self.graph_conv1 = torch_geometric.nn.conv.GraphConv(width, width).jittable()
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.graph_conv2 = torch_geometric.nn.conv.GraphConv(width, width).jittable()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch_map: Optional[Tensor] = None,
    ) -> Tensor:
        xx = self.graph_norm(x, batch_map)
        xx = self.graph_conv1(xx, edge_index, edge_weight)
        xx = self.activation(xx)
        xx = self.graph_conv2(xx, edge_index, edge_weight)
        return x + xx


class GNNSolver(nn.Module):
    def __init__(
        self,
        n_features: int,
        depth: int,
        width: int,
    ):
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        self.width = width

        self.node_embedder = nn.Linear(n_features, width)
        self.blocks = nn.ModuleList([GraphBlock(width) for _ in range(depth)])
        self.regressor = nn.Sequential(
            nn.Linear(width, width),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch_map: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.node_embedder(x)
        for block in self.blocks:
            x = block(x, edge_index, edge_weight, batch_map)
        solution = self.regressor(x)
        return solution.squeeze(dim=-1)
