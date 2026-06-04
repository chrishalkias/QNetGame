import numpy as np
import pytest


def _chain_adj(n):
    a = np.zeros((n, n))
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def _grid3_adj():
    # 3x3 grid, row-major nodes 0..8
    a = np.zeros((9, 9))
    def link(u, v): a[u, v] = a[v, u] = 1.0
    for r in range(3):
        for c in range(3):
            n = r * 3 + c
            if c < 2: link(n, n + 1)
            if r < 2: link(n, n + 3)
    return a


def test_bfs_hops_chain():
    from rl_stack.potential import bfs_hops
    d = bfs_hops(_chain_adj(10), 0)
    assert list(d) == [float(i) for i in range(10)]


def test_path_progress_disconnected_midlink():
    from rl_stack.potential import bfs_hops, path_progress
    adj = _chain_adj(10)
    d_src, d_dst = bfs_hops(adj, 0), bfs_hops(adj, 9)
    phi = path_progress(d_src, d_dst, d_src[9], [(1, 8)])   # only a mid link
    assert phi == pytest.approx(7 / 9)


def test_path_progress_chained_links():
    from rl_stack.potential import bfs_hops, path_progress
    adj = _chain_adj(10)
    d_src, d_dst = bfs_hops(adj, 0), bfs_hops(adj, 9)
    phi = path_progress(d_src, d_dst, d_src[9], [(0, 3), (3, 7)])  # chained
    assert phi == pytest.approx(7 / 9)


def test_path_progress_source_connected_matches_chain_formula():
    from rl_stack.potential import bfs_hops, path_progress
    adj = _chain_adj(10)
    d_src, d_dst = bfs_hops(adj, 0), bfs_hops(adj, 9)
    # source-connected link 0<->4: farthest f=4 -> (4-0)/9
    assert path_progress(d_src, d_dst, d_src[9], [(0, 4)]) == pytest.approx(4 / 9)


def test_path_progress_endpoints_and_empty():
    from rl_stack.potential import bfs_hops, path_progress
    adj = _chain_adj(10)
    d_src, d_dst = bfs_hops(adj, 0), bfs_hops(adj, 9)
    assert path_progress(d_src, d_dst, d_src[9], []) == 0.0          # no links
    assert path_progress(d_src, d_dst, d_src[9], [(0, 9)]) == pytest.approx(1.0)  # direct


def test_path_progress_grid_hand_case():
    from rl_stack.potential import bfs_hops, path_progress
    adj = _grid3_adj()
    d_src, d_dst = bfs_hops(adj, 0), bfs_hops(adj, 8)   # corner to corner, d_total=4
    # center node 4: d_src[4]=2, d_dst[4]=2; link 0<->4 -> span 4-0-2=2 -> 0.5
    assert path_progress(d_src, d_dst, d_src[8], [(0, 4)]) == pytest.approx(0.5)


def test_path_progress_guards():
    from rl_stack.potential import path_progress
    z = np.zeros(4)
    assert path_progress(z, z, 0.0, [(0, 1)]) == 0.0       # d_total <= 0
    assert path_progress(z, z, float("inf"), [(0, 1)]) == 0.0
