"""Distilled student Q-network: a single GraphSAGE layer over 3 local features.

One SAGEConv => 1-hop receptive field (self + nearest neighbours only). The
student only sees mean_fidelity / frac_available / can_swap, so it must decide
from internal state and immediate neighbours alone. Kept as its own class (not a
generalised QNetwork) so the teacher's conv1/conv2/conv3 state-dict keys are
never disturbed and every existing checkpoint keeps loading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# obs features the student is allowed to see (see env_wrapper get_observation):
#   1 = mean_fidelity, 3 = frac_available, 4 = can_swap
STUDENT_FEAT_IDX = [1, 3, 4]


class StudentQNetwork(nn.Module):
    """Per-node Q-network with a single GraphSAGE layer.

    Input:  Data(x=[N, 3], edge_index=[2, E])   # the STUDENT_FEAT_IDX slice
    Output: (N, n_actions) Q-values per node.
    """

    def __init__(self, node_dim: int = 3, hidden: int = 16, n_actions: int = 3):
        super().__init__()
        self.conv1 = SAGEConv(node_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, data) -> torch.Tensor:
        x = F.relu(self.conv1(data.x, data.edge_index))
        return self.head(x)          # [total_nodes_in_batch, n_actions]


def load_student(path: str, device: str = "cpu") -> StudentQNetwork:
    """Load a student checkpoint, inferring hidden/node_dim from conv1.lin_l.weight
    (mirrors rl_stack.model.load_qnet)."""
    state = torch.load(path, map_location=device, weights_only=True)
    hidden, node_dim = state["conv1.lin_l.weight"].shape
    model = StudentQNetwork(node_dim=node_dim, hidden=hidden)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":  # ponytail: round-trip + size-agnostic self-check
    import tempfile, os
    from torch_geometric.data import Data
    net = StudentQNetwork(node_dim=3, hidden=16)
    for n in (4, 9):                      # same weights, different chain length
        x = torch.randn(n, 3)
        ei = torch.tensor([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)],
                           [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]])
        assert net(Data(x=x, edge_index=ei, num_nodes=n)).shape == (n, 3)
    f = os.path.join(tempfile.mkdtemp(), "s.pth")
    torch.save(net.state_dict(), f)
    assert load_student(f).conv1.lin_l.weight.shape == (16, 3)
    print("student_model OK")
