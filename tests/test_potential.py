import itertools

import numpy as np
import pytest


def test_chain_closed_form_matches_the_bfs_formula():
    """The chain closed form must reproduce the BFS potential exactly, for every
    state. Guards the 2026-07-26 simplification: this is a REWARD function, so
    'roughly equal' is not acceptable."""
    from rl_stack.potential import path_progress
    # Full power set of link configurations for the small chains, all subsets up
    # to 3 links for the larger ones (the power set is combinatorial in N).
    for N, max_k in ((3, None), (4, None), (5, None), (6, 3), (9, 3), (12, 2)):
        d_src = np.arange(N, dtype=float)
        d_dst = np.arange(N, dtype=float)[::-1].copy()
        d_total = float(N - 1)
        pairs = list(itertools.combinations(range(N), 2))
        k_hi = len(pairs) if max_k is None else max_k
        for k in range(k_hi + 1):
            for edges in itertools.combinations(pairs, k):
                # reference: the old topology-general formula, inlined
                best = 0.0
                for a, b in edges:
                    for x, y in ((a, b), (b, a)):
                        best = max(best, d_total - d_src[x] - d_dst[y])
                ref = min(max(best, 0.0) / d_total, 1.0)
                assert path_progress(edges, N) == ref, (N, edges)


def test_path_progress_disconnected_midlink():
    from rl_stack.potential import path_progress
    phi = path_progress([(1, 8)], 10)                       # only a mid link
    assert phi == pytest.approx(7 / 9)


def test_path_progress_chained_links_uses_longest_single():
    # Single-longest-link potential: a chain of links is credited by its LONGEST
    # single link, not the union. (0,3) spans 3, (3,7) spans 4 -> 4/9. This
    # deliberately under-credits chained links: crediting their union let blobs
    # of adjacent links from background auto-entanglement game the reward.
    from rl_stack.potential import path_progress
    phi = path_progress([(0, 3), (3, 7)], 10)
    assert phi == pytest.approx(4 / 9)


def test_path_progress_adjacent_blob_not_gamed():
    # The bug fixed by reverting to single-longest-link: a full path of ADJACENT
    # links (what auto-entanglement creates for free) must NOT score high. Each
    # adjacent link spans only 1 hop -> Φ = 1/9, not 1.0.
    from rl_stack.potential import path_progress
    chain_edges = [(i, i + 1) for i in range(9)]
    assert path_progress(chain_edges, 10) == pytest.approx(1 / 9)
    # contrast: ONE swap-built long link spanning the same nodes scores 1.0
    assert path_progress([(0, 9)], 10) == pytest.approx(1.0)


def test_path_progress_source_connected_matches_chain_formula():
    from rl_stack.potential import path_progress
    # source-connected link 0<->4: farthest f=4 -> (4-0)/9
    assert path_progress([(0, 4)], 10) == pytest.approx(4 / 9)


def test_path_progress_endpoints_and_empty():
    from rl_stack.potential import path_progress
    assert path_progress([], 10) == 0.0                          # no links
    assert path_progress([(0, 9)], 10) == pytest.approx(1.0)     # direct


def test_path_progress_guards():
    from rl_stack.potential import path_progress
    assert path_progress([(0, 1)], 1) == 0.0       # degenerate chain
    assert path_progress([(0, 1)], 0) == 0.0
    assert path_progress([], 10) == 0.0            # no edges


def test_env_chain_progress_matches_potential():
    """On a chain the env potential must equal a direct path_progress call."""
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack.potential import path_progress
    env = QRNEnv(n_repeaters=6, n_ch=2, p_gen=1.0, p_swap=0.9, cutoff=20,
                 F0=1.0, channel_loss=0.0, max_steps=60,
                 rng=np.random.default_rng(1))
    env.reset()
    expected = path_progress(env._entangled_edges(), env.N)
    assert env._progress() == pytest.approx(expected)
