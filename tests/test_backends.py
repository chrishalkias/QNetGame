import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from simulator.snapshots import NodeState, Topology, _freeze


def test_freeze_makes_array_readonly():
    a = _freeze(np.array([1.0, 2.0, 3.0]))
    assert a.flags.writeable is False
    with pytest.raises(ValueError):
        a[0] = 9.0


def test_nodestate_is_frozen():
    ns = NodeState(
        node_id=0, n_ch=2, p_gen=0.8, p_swap=0.5,
        occupied=_freeze(np.zeros(2, bool)),
        partner_node=_freeze(np.full(2, -1, np.int32)),
        partner_qubit=_freeze(np.full(2, -1, np.int32)),
        fidelity=_freeze(np.zeros(2, np.float64)),
        age=_freeze(np.zeros(2, np.int32)),
        link_cutoff=_freeze(np.full(2, 20, np.int32)),
    )
    with pytest.raises(FrozenInstanceError):
        ns.node_id = 5


def test_nodestate_has_no_locked():
    """Locking machinery was removed (nothing sets it since swap/purify apply
    immediately): NodeState no longer carries a locked field."""
    ns = NodeState(
        node_id=0, n_ch=2, p_gen=0.8, p_swap=0.5,
        occupied=_freeze(np.zeros(2, bool)),
        partner_node=_freeze(np.full(2, -1, np.int32)),
        partner_qubit=_freeze(np.full(2, -1, np.int32)),
        fidelity=_freeze(np.zeros(2, np.float64)),
        age=_freeze(np.zeros(2, np.int32)),
        link_cutoff=_freeze(np.full(2, 20, np.int32)),
    )
    assert not hasattr(ns, "locked")


def test_freeze_returns_independent_copy():
    a = np.array([1.0, 2.0, 3.0])
    b = _freeze(a)
    a[0] = 999.0
    assert b[0] == 1.0  # _freeze must not alias the caller's array


def test_nodestate_autofreezes_arrays():
    occ = np.zeros(2, bool)  # writeable on the way in
    ns = NodeState(
        node_id=0, n_ch=2, p_gen=0.8, p_swap=0.5,
        occupied=occ,
        partner_node=np.full(2, -1, np.int32),
        partner_qubit=np.full(2, -1, np.int32),
        fidelity=np.zeros(2, np.float64),
        age=np.zeros(2, np.int32),
        link_cutoff=np.full(2, 20, np.int32),
    )
    assert ns.occupied.flags.writeable is False
    assert occ.flags.writeable is True  # caller's array untouched
    with pytest.raises(ValueError):
        ns.fidelity[0] = 0.5


def test_topology_autofreezes_arrays():
    adj = np.zeros((3, 3), np.float64)
    pos = np.zeros((3, 2), np.float64)
    topo = Topology(N=3, adjacency=adj, positions=pos)
    assert topo.adjacency.flags.writeable is False
    assert topo.positions.flags.writeable is False
    with pytest.raises(ValueError):
        topo.adjacency[0, 0] = 9.0


from simulator.network import build_chain, RepeaterNetwork


def _chain():
    return build_chain(3, n_ch=4, spacing=50.0, p_gen=1.0, p_swap=1.0,
                       cutoff=20, F0=0.95, channel_loss=0.0,
                       distance_dep_gen=True,
                       rng=np.random.default_rng(0))


def test_network_topology_snapshot():
    net = _chain()
    topo = net.topology()
    assert topo.N == 3
    assert topo.adjacency.shape == (3, 3)
    assert topo.positions.shape == (3, 2)
    assert topo.adjacency.flags.writeable is False


def test_node_state_reflects_entanglement():
    net = _chain()
    net.entangle(0, 1)
    ns = net.node_state(0)
    assert ns.n_ch == 4
    assert ns.occupied.any()
    qi = int(np.flatnonzero(ns.occupied)[0])
    assert int(ns.partner_node[qi]) == 1
    assert ns.fidelity[qi] > 0.0
    assert ns.occupied.flags.writeable is False


def test_node_state_free_qubit_zero_fidelity():
    net = _chain()
    ns = net.node_state(2)
    assert not ns.occupied.any()
    assert float(ns.fidelity.max()) == 0.0


def test_swap_applies_immediately():
    net = _chain()
    net.entangle(0, 1)
    net.entangle(1, 2)
    res = net.swap(1)      # applies immediately, no deferral queue
    assert res["success"]
    assert not hasattr(net, "pending_events")


from simulator.network import build_network, _sample_matched_uniform


def test_build_network_builds_chain():
    net = build_network("chain", n_repeaters=4, rng=np.random.default_rng(1))
    assert isinstance(net, RepeaterNetwork)
    assert net.topology().N == 4


def test_build_network_rejects_unknown_topology():
    with pytest.raises(ValueError):
        build_network("quantum_magic")


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
    net = build_network("chain", n_repeaters=6,
                        p_gen=0.7, p_swap=0.7, p_swap_std=0.18,
                        rng=np.random.default_rng(3))
    ps = np.array([rep.p_swap for rep in net.repeaters])
    assert ps.std() > 0.0          # inhomogeneous
    assert ps.min() >= 0.05 and ps.max() <= 1.0


def test_build_network_std_zero_is_homogeneous():
    net = build_network("chain", n_repeaters=6,
                        p_gen=0.7, p_swap=0.65,
                        rng=np.random.default_rng(4))
    assert all(rep.p_gen == 0.7 for rep in net.repeaters)
    assert all(rep.p_swap == 0.65 for rep in net.repeaters)


def test_node_state_exposes_link_cutoff():
    net = _chain()
    net.entangle(0, 1)
    ns = net.node_state(0)
    assert ns.link_cutoff.shape == (ns.n_ch,)
    assert ns.link_cutoff.flags.writeable is False
    qi = int(np.flatnonzero(ns.occupied)[0])
    assert ns.link_cutoff[qi] >= 1


def test_observation_has_urgency_feature():
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack.agent import NODE_DIM
    assert NODE_DIM == 11
    env = QRNEnv(n_repeaters=4, n_ch=4, p_gen=1.0, p_swap=1.0, cutoff=20,
                 topology="chain", rng=np.random.default_rng(0))
    env.reset()
    x = env.get_observation()["x"]
    assert x.shape == (env.N, 11)
    # urgency in [0,1); fresh links -> small; empty node -> 0
    assert (x[:, 8] >= 0).all() and (x[:, 8] <= 1).all()


def test_urgency_feature_formula():
    """feat[8] must equal mean(age[occ]/link_cutoff[occ]) from NodeState (within 1e-6).

    Uses channel_loss=0.0 so effective p_gen stays at 1.0 regardless of spacing,
    then explicitly entangles (0,1) to guarantee node 0 has occupied qubits.
    Ages 5 steps (age=5 << cutoff=20, so no expiry) and checks the arithmetic.
    """
    from rl_stack.env_wrapper import QRNEnv
    env = QRNEnv(n_repeaters=3, n_ch=4, p_gen=1.0, p_swap=0.5, cutoff=20,
                 channel_loss=0.0, topology="chain", rng=np.random.default_rng(42))
    env.reset()
    # Explicitly entangle node 0 with node 1 to guarantee occupancy at node 0
    env.net.entangle(0, 1)
    ns0 = env.net.node_state(0)
    assert ns0.occupied.any(), "entangle(0,1) must give node 0 at least one occupied qubit"
    # Age 5 steps without discarding so links survive (age=5 << cutoff=20)
    for _ in range(5):
        env.net.age_links(discard_expired=False)
    ns = env.net.node_state(0)
    occ = ns.occupied
    assert occ.any(), "node 0 must still have occupied qubits after 5 aging steps"
    lc = np.maximum(ns.link_cutoff[occ], 1)
    expected = float(np.mean(ns.age[occ] / lc))
    obs_urgency = float(env.get_observation()["x"][0, 8])
    assert abs(obs_urgency - expected) < 1e-6, (
        f"urgency feat[8]={obs_urgency:.8f} != independently computed {expected:.8f} "
        f"(age={ns.age[occ].tolist()}, lc={lc.tolist()})"
    )
