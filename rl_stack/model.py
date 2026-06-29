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

    def __init__(self, node_dim: int = 9, hidden: int = 32, n_actions: int = 3):
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


def load_qnet(path: str, device: str = "cpu") -> QNetwork:
    """Load a checkpoint, inferring node_dim/hidden from conv1.lin_l.weight."""
    state = torch.load(path, map_location=device, weights_only=True)
    hidden, node_dim = state["conv1.lin_l.weight"].shape
    model = QNetwork(node_dim=node_dim, hidden=hidden)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":  # ponytail: round-trip self-check
    import tempfile, os
    net = QNetwork(node_dim=7, hidden=16)
    f = os.path.join(tempfile.mkdtemp(), "p.pth")
    torch.save(net.state_dict(), f)
    loaded = load_qnet(f)
    assert loaded.conv1.lin_l.weight.shape == (16, 7)
    assert not loaded.training
    print("load_qnet OK")
