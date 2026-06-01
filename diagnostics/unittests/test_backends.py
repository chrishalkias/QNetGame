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
