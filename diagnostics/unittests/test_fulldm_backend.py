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


def test_fulldm_decoherence_matches_analytic_decay():
    cutoff = 20
    be = _fulldm(N=3, n_ch=4, F0=1.0, channel_loss=0.0, p_gen=1.0, cutoff=cutoff)
    be.entangle(0, 1)
    from quantum_repeater_sim.repeater import fidelity_to_werner
    for k in range(1, 11):
        be.advance()
        f = float(be.node_state(0).fidelity.max())
        w_meas = fidelity_to_werner(f)
        # calibration #2: real-qubit Werner decays as exp(-k/cutoff)
        assert abs(w_meas - np.exp(-k / cutoff)) < 1e-6


def test_fulldm_link_expires_at_cutoff():
    be = _fulldm(N=3, n_ch=4, F0=1.0, channel_loss=0.0, p_gen=1.0, cutoff=3)
    be.entangle(0, 1)
    for _ in range(3):
        be.advance()
    assert not be.node_state(0).occupied.any()
    assert not be.node_state(1).occupied.any()


def test_fulldm_swap_perfect_gives_unit_e2e_fidelity():
    # huge cutoff -> the one resolution-tick decoherence is negligible, so a
    # perfect chain yields a near-perfect e2e link (validates the BSM + Pauli
    # correction mapping without tripping on per-tick decoherence).
    be = _fulldm(N=3, n_ch=4, F0=1.0, channel_loss=0.0, p_gen=1.0, p_swap=1.0,
                 cutoff=100000, dt_seconds=0.0)
    be.entangle(0, 1)
    be.entangle(1, 2)
    res = be.swap(1)
    assert res["success"] is True
    be.advance()
    ns0 = be.node_state(0)
    q = int(np.flatnonzero(ns0.occupied)[0])
    assert int(ns0.partner_node[q]) == 2
    # calibration #3: perfect inputs -> near-perfect e2e link
    assert float(ns0.fidelity[q]) > 0.999
    assert be.node_state(1).occupied.sum() == 0   # local qubits consumed


def test_fulldm_swap_product_rule():
    # swap immediately (age 0, no decoherence yet) -> e2e Werner == product of the
    # two link Werners. cutoff huge so the single resolution-tick decoherence is
    # negligible. Exact product rule for depolarizing Werner inputs.
    be = _fulldm(N=3, n_ch=4, F0=0.9, channel_loss=0.0, p_gen=1.0, p_swap=1.0,
                 cutoff=100000, dt_seconds=0.0)
    be.entangle(0, 1); be.entangle(1, 2)
    from quantum_repeater_sim.repeater import fidelity_to_werner
    w_left = fidelity_to_werner(float(be.node_state(0).fidelity.max()))
    w_right = fidelity_to_werner(float(be.node_state(2).fidelity.max()))
    be.swap(1); be.advance()
    ns0 = be.node_state(0)
    q = int(np.flatnonzero(ns0.occupied)[0])
    w_e2e = fidelity_to_werner(float(ns0.fidelity[q]))
    assert abs(w_e2e - w_left * w_right) < 1e-3
