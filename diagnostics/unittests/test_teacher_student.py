"""Structural guards for the teacher-student distillation feature."""
import os
import tempfile

import numpy as np
import torch
from torch_geometric.data import Data

from rl_stack.teacher_student.student_model import (
    StudentQNetwork, load_student, STUDENT_FEAT_IDX)
from rl_stack.teacher_student.distill import masked_mse_loss, student_policy_fn


def _chain_edge_index(n):
    fwd = list(range(n - 1))
    return torch.tensor([fwd + [i + 1 for i in fwd], [i + 1 for i in fwd] + fwd])


def test_student_forward_shape_and_size_agnostic():
    net = StudentQNetwork(node_dim=3, hidden=16)
    for n in (4, 9):                       # same weights, different chain length
        x = torch.randn(n, 3)
        out = net(Data(x=x, edge_index=_chain_edge_index(n), num_nodes=n))
        assert out.shape == (n, 3)


def test_student_feat_idx_is_fidelity_avail_canswap():
    # documents the exact 3 features (env_wrapper obs indices)
    assert STUDENT_FEAT_IDX == [1, 3, 4]


def test_load_student_round_trip():
    net = StudentQNetwork(node_dim=3, hidden=16)
    f = os.path.join(tempfile.mkdtemp(), "s.pth")
    torch.save(net.state_dict(), f)
    loaded = load_student(f)
    assert loaded.conv1.lin_l.weight.shape == (16, 3)
    assert not loaded.training


def test_masked_mse_zero_iff_equal_and_ignores_masked_actions():
    q_t = torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 5.0]])
    mask = torch.tensor([[True, True, False], [True, False, True]])
    # identical Q -> zero loss
    assert float(masked_mse_loss(q_t.clone(), q_t, mask)) == 0.0
    # a difference ONLY on a masked action is ignored
    q_s = q_t.clone(); q_s[0, 2] += 100.0
    assert float(masked_mse_loss(q_s, q_t, mask)) == 0.0
    # a difference on a valid action shows up
    q_s = q_t.clone(); q_s[1, 2] += 2.0
    assert float(masked_mse_loss(q_s, q_t, mask)) > 0.0


def test_student_policy_fn_respects_mask():
    from rl_stack.env_wrapper import QRNEnv
    env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=1.0, p_swap=1.0, cutoff=20,
                 topology="chain", rng=np.random.default_rng(0))
    obs = env.reset()
    fn = student_policy_fn(StudentQNetwork(node_dim=3, hidden=16))
    a = fn(env, obs)
    mask = env.get_action_mask()
    assert a.shape == (env.N,)
    for i in range(env.N):
        assert mask[i, a[i]]           # never selects a masked-out action
