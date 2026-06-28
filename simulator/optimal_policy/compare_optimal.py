"""Compare a trained agent against the exact optimal policy and swap-asap.

Reuses experiments/heatmap/optimal_baseline.py for the MDP/MC machinery. The optimal
policy only exists for n_ch=2, N<=4 (exact DP), so comparison is restricted to
that slice; other points are reported against swap-asap only."""
from __future__ import annotations
import os
import pickle
import numpy as np
from typing import Dict, Optional, Sequence

from . import report as _report


def _import_optimal_baseline():
    # experiments/ is a top-level package (repo root on PYTHONPATH). Imported
    # lazily so torch/heavy deps load only when a comparison actually runs.
    from experiments.heatmap import optimal_baseline
    return optimal_baseline


def load_optimal_pickle(policy_dir: str, N: int, n_ch: int, cutoff: int,
                        horizon: int, p_gen: float, p_swap: float) -> Optional[Dict]:
    """
    Load the optimal-policy pickle for an exact config, or None if absent.

    Raises ValueError if a file exists but its stored config disagrees (never
    silently compare against the wrong policy).
    """
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


def make_agent_fns(ckpt: str, hidden: int = 64):
    """
    Return (full_fn, swaponly_fn): two policy_fn(env, obs) closures over one
    loaded agent. `swaponly_fn` masks PURIFY off so the agent is a pure
    swap-scheduler, for an apples-to-apples comparison against the swap-only DP
    optimum (which also cannot purify).
    """
    import torch
    from rl_stack.agent import QRNAgent
    from rl_stack.env_wrapper import PURIFY

    agent = QRNAgent(hidden=hidden)
    agent.policy_net.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    def full(env, obs):
        return agent.select_actions(obs, env.get_action_mask(), training=False)

    def swaponly(env, obs):
        mask = env.get_action_mask()
        mask[:, PURIFY] = False
        return agent.select_actions(obs, mask, training=False)

    return full, swaponly


def compare_to_optimal(ckpt: Optional[str], policy_dir: str, *,
                       p_gen: float, p_swap: float, cutoff: int,
                       n_range: Sequence[int] = (),
                       mc_eps: int = 2000, horizon: int = 30,
                       compare_N: Sequence[int] = (3, 4),
                       hidden: int = 64, agent_fn=None,
                       agent_fn_swaponly=None) -> Dict:
    """
    Build the optimal-comparison report at n_ch=2 for each N in compare_N.

    The agent's (p_gen, p_swap, cutoff) regime is passed explicitly; `n_range`
    is the set of sizes the agent trained on (used only to tag each row's
    `in_distribution`). The DP optimum is purify-free, reported as
    `T_opt_swaponly`. The agent is evaluated both with its full action set
    (`T_agent`) and with PURIFY masked (`T_agent_swaponly`). `agent_fn` /
    `agent_fn_swaponly` override the policies (used in tests); otherwise both
    are built from the checkpoint at `ckpt`.
    """
    ob = _import_optimal_baseline()
    n_ch = 2  # the only exact-optimal-comparable channel count
    pg, ps = p_gen, p_swap
    trained_sizes = set(n_range)

    if agent_fn is None or agent_fn_swaponly is None:
        if ckpt is None:
            raise ValueError("compare_to_optimal needs a checkpoint path or "
                             "both agent_fn and agent_fn_swaponly")
        full, swaponly = make_agent_fns(ckpt, hidden=hidden)
        agent_fn = agent_fn or full
        agent_fn_swaponly = agent_fn_swaponly or swaponly

    rows = []
    for N in compare_N:
        in_dist = N in trained_sizes
        payload = load_optimal_pickle(policy_dir, N, n_ch, cutoff, horizon, pg, ps)

        T_agent, _ = ob.mc_eval(agent_fn, N, n_ch, pg, ps, cutoff, horizon, mc_eps)
        T_agent_so, _ = ob.mc_eval(agent_fn_swaponly, N, n_ch, pg, ps, cutoff,
                                   horizon, mc_eps)
        T_swap, _ = ob.mc_eval(ob.swap_asap_fn, N, n_ch, pg, ps, cutoff, horizon, mc_eps)

        if payload is None:
            print(f"[warn] no optimal pickle for N={N} n_ch={n_ch} "
                  f"(pg={pg} ps={ps} co={cutoff}); swap-asap only")
            rows.append(_report.gaps(N, in_dist, None, T_swap, T_agent, T_agent_so))
            continue

        acts = [np.asarray(a, dtype=int) for a in payload["acts"]]
        opt_fn = ob.optimal_policy_fn(payload["policy"], acts)
        T_opt, _ = ob.mc_eval(opt_fn, N, n_ch, pg, ps, cutoff, horizon, mc_eps)
        rows.append(_report.gaps(N, in_dist, T_opt, T_swap, T_agent, T_agent_so))

    return {
        "config": {"n_ch": n_ch, "cutoff": cutoff, "p_gen": pg,
                   "p_swap": ps, "horizon": horizon, "mc_eps": mc_eps},
        "rows": rows,
    }
