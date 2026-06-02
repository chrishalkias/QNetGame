import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from quantum_repeater_sim.backends.base import (
    PhysicsBackend, NodeState, LinkState, Topology, _freeze,
)


def test_freeze_makes_array_readonly():
    a = _freeze(np.array([1.0, 2.0, 3.0]))
    assert a.flags.writeable is False
    with pytest.raises(ValueError):
        a[0] = 9.0


def test_nodestate_is_frozen():
    ns = NodeState(
        node_id=0, n_ch=2, p_gen=0.8, p_swap=0.5,
        occupied=_freeze(np.zeros(2, bool)),
        locked=_freeze(np.zeros(2, bool)),
        partner_node=_freeze(np.full(2, -1, np.int32)),
        partner_qubit=_freeze(np.full(2, -1, np.int32)),
        fidelity=_freeze(np.zeros(2, np.float64)),
        age=_freeze(np.zeros(2, np.int32)),
    )
    with pytest.raises(FrozenInstanceError):
        ns.node_id = 5


def test_physicsbackend_is_abstract():
    with pytest.raises(TypeError):
        PhysicsBackend()


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
        locked=np.zeros(2, bool),
        partner_node=np.full(2, -1, np.int32),
        partner_qubit=np.full(2, -1, np.int32),
        fidelity=np.zeros(2, np.float64),
        age=np.zeros(2, np.int32),
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


from quantum_repeater_sim.backends.legacy import LegacyBackend
from quantum_repeater_sim.network import build_chain


def _legacy_chain():
    net = build_chain(3, n_ch=4, spacing=50.0, p_gen=1.0, p_swap=1.0,
                      cutoff=20, F0=0.95, channel_loss=0.0,
                      dt_seconds=1e-4, distance_dep_gen=True,
                      rng=np.random.default_rng(0))
    return LegacyBackend(net)


def test_legacy_topology():
    be = _legacy_chain()
    topo = be.topology()
    assert topo.N == 3
    assert topo.adjacency.shape == (3, 3)
    assert topo.positions.shape == (3, 2)
    assert topo.adjacency.flags.writeable is False


def test_legacy_node_state_reflects_entanglement():
    be = _legacy_chain()
    be.entangle(0, 1)
    ns = be.node_state(0)
    assert ns.n_ch == 4
    assert ns.occupied.any()
    qi = int(np.flatnonzero(ns.occupied)[0])
    assert int(ns.partner_node[qi]) == 1
    assert ns.fidelity[qi] > 0.0
    assert ns.occupied.flags.writeable is False


def test_legacy_node_state_free_qubit_zero_fidelity():
    be = _legacy_chain()
    ns = be.node_state(2)
    assert not ns.occupied.any()
    assert float(ns.fidelity.max()) == 0.0


def test_legacy_advance_returns_contract_dict():
    be = _legacy_chain()
    out = be.advance()
    assert set(out) == {"expired", "resolved", "pending", "time"}
    assert out["time"] == 1.0


def test_legacy_pending_increments_on_deferred_swap():
    be = _legacy_chain()
    be.entangle(0, 1)
    be.entangle(1, 2)
    be.swap(1)            # nonzero channel delay -> deferred
    assert be.n_pending >= 1
