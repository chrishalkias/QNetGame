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


def test_env_grid_gives_nonzero_shaping():
    """The bug being fixed: grids previously got shaping=0. After reset
    auto-entangles, the stored potential must be > 0 on a grid."""
    import numpy as np
    from rl_stack.env_wrapper import QRNEnv
    env = QRNEnv(n_repeaters=3, n_ch=2, p_gen=1.0, p_swap=0.9, cutoff=20,
                 F0=1.0, channel_loss=0.0, dt_seconds=0.0, max_steps=60,
                 topology="grid", rng=np.random.default_rng(0))
    env.reset()
    assert env._progress() > 0.0
    assert 0.0 <= env._phi <= 1.0


def test_env_chain_progress_matches_potential():
    """On a chain the new potential must equal a direct path_progress call."""
    import numpy as np
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack.potential import bfs_hops, path_progress
    env = QRNEnv(n_repeaters=6, n_ch=2, p_gen=1.0, p_swap=0.9, cutoff=20,
                 F0=1.0, channel_loss=0.0, dt_seconds=0.0, max_steps=60,
                 topology="chain", rng=np.random.default_rng(1))
    env.reset()
    adj = env._topo.adjacency
    d_src, d_dst = bfs_hops(adj, env.source), bfs_hops(adj, env.dest)
    expected = path_progress(d_src, d_dst, d_src[env.dest], env._entangled_edges())
    assert env._progress() == pytest.approx(expected)


def test_phase2_config():
    from game.phases import PHASE2, PhaseConfig
    assert isinstance(PHASE2, PhaseConfig)
    assert PHASE2.topology == "grid"
    assert tuple(PHASE2.n_range) == (3, 4)
    assert 2 in PHASE2.n_ch
    assert PHASE2.backend == "legacy"


def test_grid_eval_runs(tmp_path):
    """grid_eval compares an injected policy vs swap-asap on small grids and
    returns a well-formed report (uses swap_asap as the 'agent' to avoid torch)."""
    import numpy as np
    from rl_stack import strategies
    from game.grid_eval import evaluate_on_grids
    agent_fn = lambda env, obs: strategies.swap_asap(env)
    report = evaluate_on_grids(agent_fn, grid_sides=(3,), n_ch=2,
                               p_gen=0.9, p_swap=0.9, cutoff=20, max_steps=40,
                               n_episodes=20, seed=0)
    assert "rows" in report and len(report["rows"]) == 1
    r = report["rows"][0]
    assert r["grid"] == 3
    assert "T_agent" in r and "T_swap_asap" in r and "agent_beats_swap_pct" in r


def test_run_phase2_smoke(tmp_path):
    import json
    from game.run_phase2 import main
    save_dir = tmp_path / "p2"
    main(["--episodes", "60", "--max_steps", "12", "--save_dir", str(save_dir),
          "--eval_episodes", "20", "--probe_episodes", "20"])
    assert (save_dir / "policy.pth").is_file()
    out = save_dir / "grid_eval.json"
    assert out.is_file()
    rep = json.loads(out.read_text())
    assert len(rep["rows"]) >= 1 and "agent_beats_swap_pct" in rep["rows"][0]
