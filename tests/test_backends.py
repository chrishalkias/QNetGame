import numpy as np
import pytest

from simulator.network import build_chain, RepeaterNetwork
from simulator.repeater import NO_PARTNER, QUBIT_OCCUPIED, werner_to_fidelity


def _chain():
    return build_chain(3, n_ch=4, spacing=50.0, p_gen=1.0, p_swap=1.0,
                       cutoff=20, F0=0.95, channel_loss=0.0,
                       distance_dep_gen=True,
                       rng=np.random.default_rng(0))


def test_network_exposes_adjacency_and_size():
    """The env reads topology straight off the engine (`net.adj` / `net.N`)
    since the Topology snapshot was deleted 2026-07-26."""
    net = _chain()
    assert net.N == 3
    assert net.adj.shape == (3, 3)
    # the chain adjacency the env's BFS/auto-entangle passes read
    assert net.adj[0, 1] == 1.0 and net.adj[1, 2] == 1.0
    assert net.adj[0, 2] == 0.0


def test_node_reflects_entanglement():
    net = _chain()
    net.entangle(0, 1)
    rep = net.node(0)
    assert rep.n_ch == 4
    occ = rep.status == QUBIT_OCCUPIED
    assert occ.any()
    qi = int(np.flatnonzero(occ)[0])
    assert int(rep.partner_repeater[qi]) == 1
    assert float(werner_to_fidelity(rep.werner_param[qi])) > 0.25


def test_node_free_qubits_carry_no_partner():
    """The live handle no longer masks fidelity to 0.0 on free qubits, so every
    consumer gates on `status == QUBIT_OCCUPIED` instead. Pin the invariant that
    makes that gate sufficient: a free qubit is never linked to anyone."""
    net = _chain()
    rep = net.node(2)
    occ = rep.status == QUBIT_OCCUPIED
    assert not occ.any()
    assert (rep.partner_repeater[~occ] == NO_PARTNER).all()


def test_node_returns_the_live_repeater():
    """node() is a zero-copy read handle onto the engine (snapshots.py was
    deleted 2026-07-26). Callers must treat it as read-only; this test pins the
    identity so a future 'defensive copy' does not silently return."""
    net = build_chain(4, n_ch=2, p_gen=1.0, p_swap=1.0, cutoff=20,
                      F0=1.0, channel_loss=0.0,
                      rng=np.random.default_rng(0))
    assert net.node(2) is net.repeaters[2]
    net.entangle(1, 2)
    # the handle reflects engine mutations without being re-fetched
    rep = net.node(2)
    before = int((rep.status == QUBIT_OCCUPIED).sum())
    net.entangle(2, 3)
    assert int((rep.status == QUBIT_OCCUPIED).sum()) == before + 1


def test_swap_applies_immediately():
    net = _chain()
    net.entangle(0, 1)
    net.entangle(1, 2)
    res = net.swap(1)      # applies immediately, no deferral queue
    assert res["success"]
    assert not hasattr(net, "pending_events")


from simulator.network import build_network, _sample_matched_uniform


def test_build_network_builds_chain():
    net = build_network(n_repeaters=4, rng=np.random.default_rng(1))
    assert isinstance(net, RepeaterNetwork)
    assert net.N == 4


def test_build_network_rejects_a_topology_argument():
    # `topology` is gone: chain is the only geometry, so there is nothing left
    # to validate. This replaces the old ValueError guard, and pins the new
    # failure mode: a stale caller must fail LOUDLY with a TypeError naming
    # build_network, never bind its string silently to another parameter.
    with pytest.raises(TypeError):
        build_network(topology="chain", n_repeaters=4)
    with pytest.raises(TypeError):
        build_network("chain", n_repeaters=4)


def test_sample_matched_uniform_std_zero_is_constant_and_drawless():
    # std=0 must broadcast the (clipped) mean and consume NO rng (stream-safe).
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    out = _sample_matched_uniform(0.7, 0.0, 5, rng_a)
    assert out.shape == (5,)
    assert np.allclose(out, 0.7)
    # rng_a must be untouched -> same next draw as the pristine rng_b
    assert rng_a.random() == rng_b.random()


def test_sample_matched_uniform_std_zero_clips_mean():
    out = _sample_matched_uniform(1.5, 0.0, 3, np.random.default_rng(0))
    assert np.allclose(out, 1.0)  # clipped to hi


def test_sample_matched_uniform_in_band_and_varies():
    mean, std = 0.6, 0.15
    out = _sample_matched_uniform(mean, std, 200, np.random.default_rng(1))
    hw = np.sqrt(3.0) * std
    assert out.min() >= max(0.05, mean - hw) - 1e-9
    assert out.max() <= min(1.0, mean + hw) + 1e-9
    assert out.std() > 0.0  # genuinely heterogeneous


def test_sample_matched_uniform_clips_to_valid_band():
    out = _sample_matched_uniform(0.95, 0.4, 500, np.random.default_rng(2))
    assert out.min() >= 0.05
    assert out.max() <= 1.0


def test_build_network_std_makes_per_repeater_params_differ():
    net = build_network(n_repeaters=6,
                        p_gen=0.7, p_swap=0.7, p_swap_std=0.18,
                        rng=np.random.default_rng(3))
    ps = np.array([rep.p_swap for rep in net.repeaters])
    assert ps.std() > 0.0          # inhomogeneous
    assert ps.min() >= 0.05 and ps.max() <= 1.0


def test_build_network_std_zero_is_homogeneous():
    net = build_network(n_repeaters=6,
                        p_gen=0.7, p_swap=0.65,
                        rng=np.random.default_rng(4))
    assert all(rep.p_gen == 0.7 for rep in net.repeaters)
    assert all(rep.p_swap == 0.65 for rep in net.repeaters)


def test_node_exposes_link_cutoff():
    net = _chain()
    net.entangle(0, 1)
    rep = net.node(0)
    # node 0 is an end node: one port of width n_ch, so n_ch qubits total
    assert rep.link_cutoff.shape == (rep.n_left + rep.n_right,)
    occ = rep.status == QUBIT_OCCUPIED
    qi = int(np.flatnonzero(occ)[0])
    assert rep.link_cutoff[qi] >= 1


def test_observation_has_normalized_age_feature():
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack.agent import NODE_DIM
    assert NODE_DIM == 8
    env = QRNEnv(n_repeaters=4, n_ch=4, p_gen=1.0, p_swap=1.0, cutoff=20,
                 rng=np.random.default_rng(0))
    env.reset()
    x = env.get_observation()["x"]
    assert x.shape == (env.N, 8)
    # normalized age in [0,1); fresh links -> small; empty node -> 0
    assert (x[:, 5] >= 0).all() and (x[:, 5] <= 1).all()


def test_normalized_age_feature_formula():
    """feat[5] must equal mean(age[occ]/link_cutoff[occ]) off the engine (within 1e-6).

    Uses channel_loss=0.0 so effective p_gen stays at 1.0 regardless of spacing,
    then explicitly entangles (0,1) to guarantee node 0 has occupied qubits.
    Ages 5 steps (age=5 << cutoff=20, so no expiry) and checks the arithmetic.
    """
    from rl_stack.env_wrapper import QRNEnv
    env = QRNEnv(n_repeaters=3, n_ch=4, p_gen=1.0, p_swap=0.5, cutoff=20,
                 channel_loss=0.0, rng=np.random.default_rng(42))
    env.reset()
    # Explicitly entangle node 0 with node 1 to guarantee occupancy at node 0
    env.net.entangle(0, 1)
    rep = env.net.node(0)
    assert (rep.status == QUBIT_OCCUPIED).any(), \
        "entangle(0,1) must give node 0 at least one occupied qubit"
    # Age 5 steps without discarding so links survive (age=5 << cutoff=20)
    for _ in range(5):
        env.net.age_links(discard_expired=False)
    # rep is the LIVE repeater, so it already reflects the aging above
    occ = rep.status == QUBIT_OCCUPIED
    assert occ.any(), "node 0 must still have occupied qubits after 5 aging steps"
    lc = np.maximum(rep.link_cutoff[occ], 1)
    ages = rep.age[occ].copy()
    expected = float(np.mean(ages / lc))
    obs_age = float(env.get_observation()["x"][0, 5])
    assert abs(obs_age - expected) < 1e-6, (
        f"normalized_age feat[5]={obs_age:.8f} != independently computed {expected:.8f} "
        f"(age={ages.tolist()}, lc={lc.tolist()})"
    )
