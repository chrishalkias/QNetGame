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


from quantum_repeater_sim.backends import make_backend


def test_factory_builds_fulldm_chain():
    be = make_backend("netsquid", topology="chain", n_repeaters=4,
                      fidelity_mode="full_dm", rng=np.random.default_rng(0))
    assert isinstance(be, FullDMBackend)
    assert be.topology().N == 4


def test_factory_fulldm_rejects_non_chain():
    with pytest.raises(NotImplementedError):
        make_backend("netsquid", topology="grid", fidelity_mode="full_dm")


def test_qrnenv_constructs_on_fulldm():
    from rl_stack.env_wrapper import QRNEnv
    env = QRNEnv(n_repeaters=3, topology="chain", backend="netsquid",
                 fidelity_mode="full_dm", dt_seconds=0.0,
                 rng=np.random.default_rng(1))
    obs = env.reset()
    assert obs["x"].shape == (env.N, 10)


def test_fulldm_env_rollout_is_valid():
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack.strategies import swap_asap
    env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=0.9, p_swap=0.7, cutoff=20,
                 max_steps=40, topology="chain", backend="netsquid",
                 fidelity_mode="full_dm", dt_seconds=0.0,
                 rng=np.random.default_rng(2024))
    obs = env.reset()
    assert obs["x"].shape == (env.N, 10)
    for _ in range(40):
        mask = env.get_action_mask()
        a = swap_asap(env)
        for i in range(env.N):
            assert mask[i, a[i]]
        obs, r, done, info = env.step(a)
        assert np.isfinite(r)
        assert np.all(obs["x"] >= -1e-6) and np.all(obs["x"] <= 1.0 + 1e-6)
        if done:
            break


def test_fulldm_e2e_reachable_perfect_chain():
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack.strategies import swap_asap
    env = QRNEnv(n_repeaters=3, n_ch=4, p_gen=1.0, p_swap=1.0, cutoff=50,
                 max_steps=80, topology="chain", backend="netsquid",
                 fidelity_mode="full_dm", dt_seconds=0.0,
                 rng=np.random.default_rng(7))
    env.reset()
    ok = False
    for _ in range(80):
        _, _, done, info = env.step(swap_asap(env))
        if done and info["fidelity"] > 0.0:
            ok = True
            break
    assert ok


def test_fulldm_parity_with_analytic():
    """full_dm reproduces analytic e2e fidelity (depolarizing ⇒ Werner-exact).
    Built one engine at a time (NetSquid global sim state)."""
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack.strategies import swap_asap

    def run(mode):
        env = QRNEnv(n_repeaters=3, n_ch=4, p_gen=1.0, p_swap=1.0, cutoff=50,
                     max_steps=80, topology="chain", backend="netsquid",
                     fidelity_mode=mode, dt_seconds=0.0,
                     rng=np.random.default_rng(11))
        env.reset()
        for _ in range(80):
            _, _, done, info = env.step(swap_asap(env))
            if done:
                return info["fidelity"]
        return 0.0

    f_analytic = run("analytic")
    f_fulldm = run("full_dm")
    assert f_analytic > 0.0 and f_fulldm > 0.0     # both reach e2e
    # Same rng drives identical control flow; depolarizing keeps states Werner, so
    # the two engines agree closely. NOT required to be bit-identical: analytic's
    # inherited-age swap bookkeeping double-counts decoherence by one extra
    # exp(-max_age/cutoff) factor vs full_dm's real per-tick depolarization.
    # swap_asap swaps at age 0, so the gap is small; allow 2e-2.
    assert abs(f_fulldm - f_analytic) < 2e-2
