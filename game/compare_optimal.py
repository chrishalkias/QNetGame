"""Compare a trained agent against the exact optimal policy and swap-asap.

Reuses train-test/optimal_baseline.py for the MDP/MC machinery. The optimal
policy only exists for n_ch=2, N<=4 (exact DP), so comparison is restricted to
that slice; other points are reported against swap-asap only."""
from __future__ import annotations
import os
import sys
import pickle
import numpy as np
from typing import Dict, Optional, Sequence

from .phases import PhaseConfig
from . import report as _report


def _repo_root() -> str:
    # game/compare_optimal.py -> game/ -> repo root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_optimal_baseline():
    tt = os.path.join(_repo_root(), "train-test")
    if tt not in sys.path:
        sys.path.append(tt)
    import optimal_baseline  # noqa: E402
    return optimal_baseline


def load_optimal_pickle(policy_dir: str, N: int, n_ch: int, cutoff: int,
                        horizon: int, p_gen: float, p_swap: float) -> Optional[Dict]:
    """Load the optimal-policy pickle for an exact config, or None if absent.

    Raises ValueError if a file exists but its stored config disagrees (never
    silently compare against the wrong policy)."""
    fname = (f"optimal_policy_N{N}_ch{n_ch}_co{cutoff}_h{horizon}"
             f"_pg{p_gen:.2f}_ps{p_swap:.2f}.pkl")
    path = os.path.join(policy_dir, fname)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    want = dict(N=N, n_ch=n_ch, cutoff=cutoff, horizon=horizon,
                p_gen=p_gen, p_swap=p_swap)
    cfg = payload.get("config", {})
    for k, v in want.items():
        if cfg.get(k) != v:
            raise ValueError(f"pickle {fname} config mismatch on {k}: "
                             f"{cfg.get(k)!r} != {v!r}")
    return payload


def compare_to_optimal(ckpt: Optional[str], cfg: PhaseConfig, policy_dir: str,
                       mc_eps: int = 2000, horizon: int = 30,
                       compare_N: Sequence[int] = (3, 4),
                       hidden: int = 64, agent_fn=None) -> Dict:
    """Build the optimal-comparison report at n_ch=2 for each N in compare_N.

    `agent_fn` overrides the agent policy (used in tests); otherwise the trained
    checkpoint at `ckpt` is loaded via optimal_baseline.make_agent_fn."""
    ob = _import_optimal_baseline()
    n_ch = 2  # the only exact-optimal-comparable channel count
    pg, ps, cutoff = cfg.p_gen, cfg.p_swap, cfg.cutoff
    trained_sizes = set(cfg.n_range)

    if agent_fn is None:
        if ckpt is None:
            raise ValueError("compare_to_optimal needs a checkpoint path or agent_fn")
        agent_fn = ob.make_agent_fn(ckpt, hidden=hidden)

    rows = []
    for N in compare_N:
        in_dist = N in trained_sizes
        payload = load_optimal_pickle(policy_dir, N, n_ch, cutoff, horizon, pg, ps)

        T_agent, _ = ob.mc_eval(agent_fn, N, n_ch, pg, ps, cutoff, horizon, mc_eps)
        T_swap, _ = ob.mc_eval(ob.swap_asap_fn, N, n_ch, pg, ps, cutoff, horizon, mc_eps)

        if payload is None:
            print(f"[warn] no optimal pickle for N={N} n_ch={n_ch} "
                  f"(pg={pg} ps={ps} co={cutoff}); swap-asap only")
            rows.append(_report.gaps(N, in_dist, None, T_swap, T_agent))
            continue

        acts = [np.asarray(a, dtype=int) for a in payload["acts"]]
        opt_fn = ob.optimal_policy_fn(payload["policy"], acts)
        T_opt, _ = ob.mc_eval(opt_fn, N, n_ch, pg, ps, cutoff, horizon, mc_eps)
        rows.append(_report.gaps(N, in_dist, T_opt, T_swap, T_agent))

    return {
        "config": {"n_ch": n_ch, "cutoff": cutoff, "p_gen": pg,
                   "p_swap": ps, "horizon": horizon, "mc_eps": mc_eps},
        "rows": rows,
    }
