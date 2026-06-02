import numpy as np
import pytest

from quantum_repeater_sim.backends.netsquid.timing import SimClock, TICK_NS


def test_delay_ticks_zero_when_dt_zero():
    clk = SimClock(c_fiber=200_000.0, dt_seconds=0.0)
    assert clk.delay_ticks(123.4) == 0
    assert clk.delay_ticks(0.0) == 0


def test_delay_ticks_matches_legacy_formula():
    clk = SimClock(c_fiber=200_000.0, dt_seconds=1e-4)
    # ceil(d / (c * dt)) = ceil(50 / (200000 * 1e-4)) = ceil(50/20) = 3
    assert clk.delay_ticks(50.0) == 3
    assert clk.delay_ticks(0.0) == 0


def test_callback_fires_after_correct_number_of_advances():
    clk = SimClock(c_fiber=200_000.0, dt_seconds=1e-4)
    clk.reset()
    fired = []
    clk.schedule(delay_ticks=2, callback=lambda: fired.append(clk.tick))
    clk.advance()                      # tick 1 — not yet
    assert fired == []
    clk.advance()                      # tick 2 — fires
    assert fired == [2]
    assert clk.tick == 2


def test_delay_zero_fires_on_next_advance():
    clk = SimClock(c_fiber=200_000.0, dt_seconds=0.0)
    clk.reset()
    fired = []
    clk.schedule(delay_ticks=0, callback=lambda: fired.append(True))
    clk.advance()                      # resolves same/next tick
    assert fired == [True]


def test_reset_clears_pending_and_tick():
    clk = SimClock(c_fiber=200_000.0, dt_seconds=1e-4)
    clk.reset()
    fired = []
    clk.schedule(delay_ticks=5, callback=lambda: fired.append(True))
    assert clk.n_pending == 1
    clk.advance()
    clk.reset()
    assert clk.n_pending == 0
    assert clk.tick == 0
    # orphaned callback from before reset must never fire
    for _ in range(8):
        clk.advance()
    assert fired == []


from quantum_repeater_sim.backends.netsquid.backend import NetSquidBackend
from quantum_repeater_sim.backends.base import NodeState, Topology


def _chain_backend(N=3, n_ch=4, **kw):
    params = dict(n_ch=n_ch, spacing=50.0, p_gen=1.0, p_swap=1.0, cutoff=20,
                  F0=0.95, channel_loss=0.0, dt_seconds=0.0,
                  distance_dep_gen=True, rng=np.random.default_rng(0))
    params.update(kw)
    return NetSquidBackend(N=N, **params)


def test_backend_topology_chain():
    be = _chain_backend(N=3)
    topo = be.topology()
    assert isinstance(topo, Topology)
    assert topo.N == 3
    assert topo.adjacency.shape == (3, 3)
    assert topo.adjacency[0, 1] != 0 and topo.adjacency[1, 2] != 0
    assert topo.adjacency[0, 2] == 0
    assert topo.positions.shape == (3, 2)
    assert topo.adjacency.flags.writeable is False


def test_backend_empty_node_state():
    be = _chain_backend(N=3, n_ch=4)
    ns_ = be.node_state(0)
    assert isinstance(ns_, NodeState)
    assert ns_.n_ch == 4
    assert not ns_.occupied.any()
    assert float(ns_.fidelity.max()) == 0.0
    assert ns_.occupied.flags.writeable is False


def test_backend_time_and_pending_start_zero():
    be = _chain_backend()
    assert be.time == 0.0
    assert be.n_pending == 0
    assert be.all_links() == []


def test_backend_reset_clears_state():
    be = _chain_backend()
    be.reset()
    assert be.time == 0.0
    assert be.n_pending == 0


def test_entangle_creates_link_with_expected_fidelity():
    be = _chain_backend(N=3, n_ch=4, F0=0.9, channel_loss=0.0, p_gen=1.0)
    res = be.entangle(0, 1)
    assert res["success"] is True
    ns0, ns1 = be.node_state(0), be.node_state(1)
    q0 = int(np.flatnonzero(ns0.occupied)[0])
    q1 = int(np.flatnonzero(ns1.occupied)[0])
    assert int(ns0.partner_node[q0]) == 1
    assert int(ns1.partner_node[q1]) == 0
    assert abs(float(ns0.fidelity[q0]) - 0.9) < 1e-9


def test_entangle_non_adjacent_fails():
    be = _chain_backend(N=3)
    assert be.entangle(0, 2)["success"] is False


def test_advance_ages_and_decoheres_link():
    be = _chain_backend(N=3, n_ch=4, F0=1.0, channel_loss=0.0,
                        p_gen=1.0, cutoff=10)
    be.entangle(0, 1)
    f0 = float(be.node_state(0).fidelity.max())
    be.advance()
    f1 = float(be.node_state(0).fidelity.max())
    assert f1 < f0
    assert be.time == 1.0


def test_advance_expires_link_at_cutoff():
    be = _chain_backend(N=3, n_ch=4, F0=1.0, channel_loss=0.0,
                        p_gen=1.0, cutoff=3)
    be.entangle(0, 1)
    for _ in range(3):
        be.advance()
    assert not be.node_state(0).occupied.any()
    assert not be.node_state(1).occupied.any()


def test_swap_perfect_links_resolves_to_e2e():
    # 3-node chain, perfect gen/swap, no loss, no CC delay (dt=0)
    be = _chain_backend(N=3, n_ch=4, F0=1.0, channel_loss=0.0,
                        p_gen=1.0, p_swap=1.0, cutoff=50, dt_seconds=0.0)
    be.entangle(0, 1)
    be.entangle(1, 2)
    res = be.swap(1)
    assert res["success"] is True
    be.advance()   # resolution fires (dt=0 -> next advance)
    ns0 = be.node_state(0)
    partners = [int(p) for p in ns0.partner_node[ns0.occupied]]
    assert 2 in partners
    assert be.node_state(1).occupied.sum() == 0


def test_swap_insufficient_qubits_fails():
    be = _chain_backend(N=3)
    assert be.swap(1)["success"] is False


def test_swap_product_rule_fidelity():
    be = _chain_backend(N=3, n_ch=4, F0=1.0, channel_loss=0.0,
                        p_gen=1.0, p_swap=1.0, cutoff=1000, dt_seconds=0.0)
    be.entangle(0, 1); be.entangle(1, 2)
    occ1 = np.flatnonzero(be._occupied[1])
    p_a = be._current_werner(1, int(occ1[0]))
    p_b = be._current_werner(1, int(occ1[1]))
    be.swap(1); be.advance()
    ns0 = be.node_state(0)
    q = int(np.flatnonzero(ns0.occupied)[0])
    # Product rule applies to the *initial* werner (p0); current fidelity is
    # decohered by the one tick the locked remote qubits age during the CC
    # delay before resolution (matches legacy network age_links/_resolve_swap).
    assert abs(float(be._p0[0, q]) - p_a * p_b) < 1e-6


def test_swap_stale_remote_expiry_is_safe():
    # remote link expires during the CC delay -> resolution must drop cleanly
    be = _chain_backend(N=3, n_ch=4, F0=1.0, channel_loss=0.0, p_gen=1.0,
                        p_swap=1.0, cutoff=2, spacing=5000.0, dt_seconds=1e-4)
    be.entangle(0, 1); be.entangle(1, 2)
    res = be.swap(1)
    assert res["success"] is True            # deferred (nonzero CC delay)
    assert be.n_pending >= 1
    for _ in range(6):                        # outlast cutoff and the delay
        be.advance()
    ns0 = be.node_state(0)
    assert 2 not in [int(p) for p in ns0.partner_node[ns0.occupied]]


from quantum_repeater_sim.backends import make_backend


def test_factory_builds_netsquid_chain_analytic():
    be = make_backend("netsquid", topology="chain", n_repeaters=4,
                      fidelity_mode="analytic", rng=np.random.default_rng(0))
    assert isinstance(be, NetSquidBackend)
    assert be.topology().N == 4


def test_factory_netsquid_rejects_non_chain():
    with pytest.raises(NotImplementedError):
        make_backend("netsquid", topology="grid", n_repeaters=3)


def test_factory_netsquid_rejects_full_dm():
    with pytest.raises(NotImplementedError):
        make_backend("netsquid", topology="chain", fidelity_mode="full_dm")


def test_qrnenv_constructs_on_netsquid():
    from rl_stack.env_wrapper import QRNEnv
    env = QRNEnv(n_repeaters=4, topology="chain", backend="netsquid",
                 dt_seconds=0.0, rng=np.random.default_rng(1))
    obs = env.reset()
    assert obs["x"].shape == (env.N, 8)
