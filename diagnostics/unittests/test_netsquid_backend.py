import math
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
    clk.schedule(delay_ticks=5, callback=lambda: None)
    assert clk.n_pending == 1
    clk.advance()
    clk.reset()
    assert clk.n_pending == 0
    assert clk.tick == 0
