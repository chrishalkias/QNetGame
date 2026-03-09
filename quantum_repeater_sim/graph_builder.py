"""Build hierarchical heterogeneous graph from a RepeaterNetwork."""

from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from .network import RepeaterNetwork
from .repeater import NO_PARTNER, werner_to_fidelity
import torch
from torch_geometric.data import HeteroData as _HeteroData


def _mk(arr: np.ndarray, dnp, dtorch=None):
    """Transform an array into torch tensor with specified dtype"""
    a = np.asarray(arr, dtype=dnp)
    return torch.tensor(a, dtype=dtorch)


def network_to_heterodata(net: RepeaterNetwork):
    """
    Creates the GNN representation of the `RepeaterNetwork`

    """
    N, n_ch = net.N, net.repeaters[0].n_ch
    data = _HeteroData()
    _f = torch.float32
    _i = torch.long
    _b = torch.bool

    # ---Repeater features (N, 6)
    data["repeater"].x = _mk(
        np.stack([r.feature_vector() for r in net.repeaters]), np.float32, _f)
    # ---Qubit features (N*n_ch, 6) — includes locked column
    data["qubit"].x = _mk(
        np.concatenate([r.qubit_features() for r in net.repeaters]), np.float32, _f)

    # ---Adjacency
    s, d = np.nonzero(net.adj)
    if len(s):
        data["repeater","adjacent","repeater"].edge_index = _mk(np.stack([s,d]), np.int64, _i)
        dd = net._dist_matrix[s, d]
        data["repeater","adjacent","repeater"].edge_attr = _mk(
            (dd / max(dd.max(), 1e-30)).reshape(-1,1), np.float32, _f)
    else:
        data["repeater","adjacent","repeater"].edge_index = _mk(np.zeros((2,0)), np.int64, _i)

    # ---Ownership
    hs = np.repeat(np.arange(N, dtype=np.int64), n_ch)
    hd = np.arange(N * n_ch, dtype=np.int64)
    data["repeater","has","qubit"].edge_index = _mk(np.stack([hs, hd]), np.int64, _i)
    data["qubit","belongs_to","repeater"].edge_index = _mk(np.stack([hd, hs]), np.int64, _i)

    # ---Entanglement
    es, ed, ef = [], [], []
    for rep in net.repeaters:
        for qi in rep.occupied_indices():
            pr, pq = int(rep.partner_repeater[qi]), int(rep.partner_qubit[qi])
            if pr == NO_PARTNER: continue
            es.append(rep.rid * n_ch + int(qi))
            ed.append(pr * n_ch + pq)
            ef.append(werner_to_fidelity(rep.werner_param[qi]))
    if es:
        data["qubit","entangled","qubit"].edge_index = _mk(np.stack([es,ed]), np.int64, _i)
        data["qubit","entangled","qubit"].edge_attr = _mk(
            np.array(ef, dtype=np.float32).reshape(-1,1), np.float32, _f)
    else:
        data["qubit","entangled","qubit"].edge_index = _mk(np.zeros((2,0)), np.int64, _i)
        data["qubit","entangled","qubit"].edge_attr = _mk(np.zeros((0,1)), np.float32, _f)

    # ---Action masks
    data["repeater"].swap_mask = _mk(net.action_mask_swap(), np.bool_, _b)
    em = net.action_mask_entangle(); ms, md = np.nonzero(np.triu(em))
    data["entangle_mask"] = _mk(np.stack([ms, md]), np.int64, _i)
    pm = net.action_mask_purify(); ps, pd = np.nonzero(np.triu(pm))
    data["purify_mask"] = _mk(np.stack([ps, pd]), np.int64, _i)

    # Network-level scalar features (stored on repeater node type)
    data["repeater"].network_state = _mk(
        np.array([net.time_step, len(net.pending_events)], dtype=np.float32),
        np.float32, _f)

    return data