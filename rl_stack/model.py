"""GNN that outputs per-node Q-values for the 3 repeater actions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class QNetwork(nn.Module):
    """Per-node Q-network using GraphSAGE message passing.

    Input:  Data(x=[N, node_dim], edge_index=[2, E])
    Output: (N, n_actions) Q-values per node.
    """

    def __init__(self, node_dim: int = 8, hidden: int = 32, n_actions: int = 3):
        super().__init__()
        self.conv1 = SAGEConv(node_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, data) -> torch.Tensor:
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = F.relu(self.conv3(x, ei))
        return self.head(x)          # [total_nodes_in_batch, n_actions]