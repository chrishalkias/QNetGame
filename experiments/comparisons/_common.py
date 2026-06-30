"""Shared helpers for the experiments/comparisons sweeps.

All evals reuse optimal_baseline.mc_eval (delivery time T = avg steps to
termination, censored at the horizon) and the same three policies, so every
comparison is a paired, identically-configured measurement.
"""
from __future__ import annotations
import json, math, os
import numpy as np


def build_policies(ckpt, hidden=64):
    """{name: policy_fn(env, obs)} for agent / swap-ASAP / purify-then-swap."""
    from experiments.heatmap import optimal_baseline as ob
    from rl_stack import strategies
    return {
        "agent":       ob.make_agent_fn(ckpt, hidden=hidden),
        "swap_asap":   lambda env, obs: strategies.swap_asap(env),
        "purify_swap": lambda env, obs: strategies.purify_then_swap(env),
    }


def eval_T(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps):
    """(mean T, standard error) for one policy at one config."""
    from experiments.heatmap import optimal_baseline as ob
    T, sd = ob.mc_eval(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps)
    return float(T), float(sd / math.sqrt(mc_eps))


def action_fractions(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps, seed=42):
    """Fraction of [NOOP, SWAP, PURIFY] among interior-node decisions over
    mc_eps greedy rollouts (source/dest excluded; the three sum to 1)."""
    from rl_stack.env_wrapper import QRNEnv
    rng = np.random.default_rng(seed)
    counts = np.zeros(3, dtype=np.int64)
    for _ in range(mc_eps):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     F0=1.0, channel_loss=0.0, dt_seconds=0.0, max_steps=H,
                     topology="chain", rng=np.random.default_rng(int(rng.integers(2**32))))
        obs = env.reset()
        for _ in range(H):
            acts = policy_fn(env, obs)
            for i in range(env.N):
                if i not in (env.source, env.dest):
                    counts[int(acts[i])] += 1
            obs, _, done, _ = env.step(acts)
            if done:
                break
    return (counts / max(counts.sum(), 1)).tolist()   # [f_noop, f_swap, f_purify]


def save_json(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(rows, open(path, "w"), indent=2)


def load_json(path):
    return json.load(open(path))


PLOT_RC = {"font.size": 10, "figure.dpi": 150}
COLORS = {"agent": "tab:blue", "swap_asap": "tab:orange", "purify_swap": "tab:green"}
LABELS = {"agent": "Agent", "swap_asap": "Swap-ASAP", "purify_swap": "Purify-then-swap"}
ACTION_COLORS = ["#bdbdbd", "tab:blue", "tab:green"]   # noop, swap, purify
ACTION_LABELS = ["NOOP", "SWAP", "PURIFY"]


def savefig(fig, stem):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    print(f"saved -> {stem}.png / .pdf")
