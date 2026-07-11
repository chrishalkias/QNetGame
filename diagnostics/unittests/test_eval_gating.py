"""Entanglement-gated evaluation: unit tests for the F>1/2 delivery gate.

These test the gating arithmetic (which delivered episodes count toward T and
which fidelity averages) without paying for real GNN/physics rollouts: the real
QRNEnv is swapped for a tiny scripted env that delivers on the first step with a
fidelity we control per episode. The physics of the swap chain is exercised by
test_simulator.py; here we only pin down the censoring rule.

Gate under test (optimal_baseline.mc_eval):
  f_min is None -> time-to-connection (delivered iff F > 0).
  f_min = 0.5   -> time-to-entanglement (a connected-but-separable episode is
                   censored at H, exactly like a never-connected episode).
The env terminates on the FIRST connection, so a separable first delivery is a
failure for T_ent: the policy gets no second attempt.
"""
import numpy as np
import pytest

from experiments.heatmap import optimal_baseline as ob


class _ScriptedEnv:
    """Delivers on the first step with a fidelity pulled, in order, from a
    class-level list that cycles across episodes. Only implements the surface
    mc_eval touches (reset/step + an info['fidelity'])."""
    fids = [1.0]          # per-episode delivered fidelity, cycled
    _ep = 0

    def __init__(self, *a, **k):
        self._f = _ScriptedEnv.fids[_ScriptedEnv._ep % len(_ScriptedEnv.fids)]
        _ScriptedEnv._ep += 1

    def reset(self):
        return None

    def step(self, action):
        return None, 0.0, True, {"fidelity": self._f}


def _run(fids, f_min, return_stats=False, H=50, n_episodes=None):
    _ScriptedEnv.fids = list(fids)
    _ScriptedEnv._ep = 0
    n = n_episodes if n_episodes is not None else len(fids)
    return ob.mc_eval(lambda env, obs: 0, N=3, n_ch=2, p_gen=0.5, p_swap=0.5,
                      cutoff=5, H=H, n_episodes=n, f_min=f_min,
                      return_stats=return_stats)


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    monkeypatch.setattr(ob, "QRNEnv", _ScriptedEnv)
    yield


def test_connection_gate_counts_any_connection():
    # f_min=None: even a separable (F=0.4) delivery counts, T = step+1 = 1.
    T, _ = _run([0.4], f_min=None)
    assert T == pytest.approx(1.0)


def test_entanglement_gate_censors_separable_delivery():
    # F=0.4 <= 1/2 is separable: under the entanglement gate it is censored at H.
    T, _ = _run([0.4], f_min=0.5, H=50)
    assert T == pytest.approx(50.0)


def test_entanglement_gate_keeps_entangled_delivery():
    # F=0.6 > 1/2 is entangled: delivered on step 0, T = 1.
    T, _ = _run([0.6], f_min=0.5, H=50)
    assert T == pytest.approx(1.0)


def test_gated_time_averages_over_right_subset():
    # Two entangled (T=1) + two separable (censored at H=50) -> mean = 25.5.
    T, _ = _run([0.6, 0.6, 0.4, 0.4], f_min=0.5, H=50)
    assert T == pytest.approx((1 + 1 + 50 + 50) / 4.0)


def test_stats_dict_fidelity_subsets():
    stats = _run([0.6, 0.4, 0.9], f_min=0.5, H=50, return_stats=True)
    # All three connect topologically; only F=0.6 and F=0.9 are entangled.
    assert stats["conn_rate"] == pytest.approx(1.0)
    assert stats["delivery_rate"] == pytest.approx(2.0 / 3.0)
    assert stats["mean_F_conn"] == pytest.approx((0.6 + 0.4 + 0.9) / 3.0)
    assert stats["mean_F_ent"] == pytest.approx((0.6 + 0.9) / 2.0)


def test_stats_dict_no_entangled_delivery():
    stats = _run([0.3, 0.4], f_min=0.5, H=20, return_stats=True)
    assert stats["conn_rate"] == pytest.approx(1.0)
    assert stats["delivery_rate"] == pytest.approx(0.0)
    assert stats["mean_F_conn"] == pytest.approx((0.3 + 0.4) / 2.0)
    assert stats["mean_F_ent"] is None
    assert stats["T"] == pytest.approx(20.0)


def test_connection_stats_delivery_rate_matches_conn_rate():
    # Under the connection gate (f_min=None), delivery_rate == conn_rate.
    stats = _run([0.3, 0.6, 0.9], f_min=None, H=20, return_stats=True)
    assert stats["delivery_rate"] == pytest.approx(stats["conn_rate"])
    assert stats["delivery_rate"] == pytest.approx(1.0)


def test_default_return_is_mean_std_tuple():
    # Byte-compatibility for existing callers: default call returns (mean, std).
    out = ob.mc_eval(lambda env, obs: 0, N=3, n_ch=2, p_gen=0.5, p_swap=0.5,
                     cutoff=5, H=10, n_episodes=4)
    assert isinstance(out, tuple) and len(out) == 2
