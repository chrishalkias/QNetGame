import numpy as np
from rl_stack.winnability import WinnabilityCache
from rl_stack import strategies


def test_easy_cell_winnable_hard_cell_not():
    wc = WinnabilityCache(n_pilots=8, probe_steps=300, seed=0)
    # easy: high rates, generous cutoff, small chain
    assert wc.winnable(p_gen=0.95, p_swap=0.95, cutoff=40, n_repeaters=4, n_ch=4)
    # impossible: links die (cutoff=1) before any swap can assemble a 10-node e2e
    assert not wc.winnable(p_gen=0.3, p_swap=0.3, cutoff=1, n_repeaters=10, n_ch=2)


def test_results_are_cached_per_bin():
    wc = WinnabilityCache(n_pilots=4, probe_steps=200, seed=0)
    wc.winnable(p_gen=0.9, p_swap=0.9, cutoff=30, n_repeaters=4, n_ch=4)
    calls = wc.pilot_calls
    # same coarse bin again -> cache hit, no new pilots run
    wc.winnable(p_gen=0.9, p_swap=0.9, cutoff=30, n_repeaters=4, n_ch=4)
    assert wc.pilot_calls == calls


def test_ticks_not_microsteps_large_n_not_wrongly_pruned():
    # Regression for the tick/micro-step mixup: probe_steps must be read as
    # TICKS (env.steps), not micro-decisions. At N=10 (8 interior nodes) a
    # micro-step-counting pilot only covers probe_steps/8 ticks, which can
    # wrongly mark a winnable cell unwinnable. seed=5 with these params is a
    # verified divergence point (pre-fix: unwinnable; post-fix: winnable).
    wc = WinnabilityCache(n_pilots=5, probe_steps=1000, seed=5)
    assert wc.winnable(p_gen=0.4, p_swap=0.5, cutoff=25, n_repeaters=10, n_ch=2)


def test_oracle_is_purify_then_swap(monkeypatch):
    calls = {"n": 0}
    real = strategies.purify_then_swap
    def spy(env):
        calls["n"] += 1
        return real(env)
    monkeypatch.setattr(strategies, "purify_then_swap", spy)
    wc = WinnabilityCache(n_pilots=1, probe_steps=30, seed=0)
    wc.winnable(1.0, 1.0, 20, 3, 2)
    assert calls["n"] > 0
