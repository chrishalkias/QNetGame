import numpy as np
import pytest

from quantum_repeater_sim.backends.netsquid.fulldm import FullDMBackend


def _fulldm(N=3, n_ch=4, **kw):
    params = dict(n_ch=n_ch, spacing=50.0, p_gen=1.0, p_swap=1.0, cutoff=50,
                  F0=0.95, channel_loss=0.0, dt_seconds=0.0,
                  distance_dep_gen=True, rng=np.random.default_rng(0))
    params.update(kw)
    return FullDMBackend(N=N, **params)


def test_fulldm_entangle_real_fidelity():
    be = _fulldm(F0=0.9, channel_loss=0.0)
    res = be.entangle(0, 1)
    assert res["success"] is True
    ns0 = be.node_state(0)
    q0 = int(np.flatnonzero(ns0.occupied)[0])
    # calibration #1: measured real-qubit fidelity == target F0*exp(0) = 0.9
    assert abs(float(ns0.fidelity[q0]) - 0.9) < 1e-6


def test_fulldm_free_slot_zero_fidelity():
    be = _fulldm()
    ns2 = be.node_state(2)
    assert not ns2.occupied.any()
    assert float(ns2.fidelity.max()) == 0.0


def test_fulldm_reset_clears_qubits():
    be = _fulldm()
    be.entangle(0, 1)
    be.reset()
    assert not be.node_state(0).occupied.any()
    assert be.time == 0.0
