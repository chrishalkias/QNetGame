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


def test_path_progress_chained_links_uses_longest_single():
    # Single-longest-link potential: a chain of links is credited by its LONGEST
    # single link, not the union. (0,3) spans 3, (3,7) spans 4 -> 4/9. This
    # deliberately under-credits chained links: crediting their union let blobs
    # of adjacent links from background auto-entanglement game the reward.
    from rl_stack.potential import bfs_hops, path_progress
    adj = _chain_adj(10)
    d_src, d_dst = bfs_hops(adj, 0), bfs_hops(adj, 9)
    phi = path_progress(d_src, d_dst, d_src[9], [(0, 3), (3, 7)])
    assert phi == pytest.approx(4 / 9)


def test_path_progress_adjacent_blob_not_gamed():
    # The bug fixed by reverting to single-longest-link: a full path of ADJACENT
    # links (what auto-entanglement creates for free) must NOT score high. Each
    # adjacent link spans only 1 hop -> Φ = 1/9, not 1.0.
    from rl_stack.potential import bfs_hops, path_progress
    adj = _chain_adj(10)
    d_src, d_dst = bfs_hops(adj, 0), bfs_hops(adj, 9)
    chain_edges = [(i, i + 1) for i in range(9)]
    assert path_progress(d_src, d_dst, d_src[9], chain_edges) == pytest.approx(1 / 9)


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


def test_env_chain_progress_matches_potential():
    """On a chain the new potential must equal a direct path_progress call."""
    import numpy as np
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack.potential import bfs_hops, path_progress
    env = QRNEnv(n_repeaters=6, n_ch=2, p_gen=1.0, p_swap=0.9, cutoff=20,
                 F0=1.0, channel_loss=0.0, max_steps=60,
                 topology="chain", rng=np.random.default_rng(1))
    env.reset()
    adj = env.net.adj
    d_src, d_dst = bfs_hops(adj, env.source), bfs_hops(adj, env.dest)
    expected = path_progress(d_src, d_dst, d_src[env.dest], env._entangled_edges())
    assert env._progress() == pytest.approx(expected)
