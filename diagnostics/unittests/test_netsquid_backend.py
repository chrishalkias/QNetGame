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
