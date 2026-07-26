"""
--------------------------------------------------------------------------------
Shared helpers for the experiments/comparisons sweeps.

Every eval goes through mc_eval.mc_eval_stats (delivery time T, censored at
the horizon) and the same three policies, so every comparison is a paired,
identically-configured measurement. Delivery is topological (source connects
to dest); the cutoff already bounds delivered-link decoherence, so no
fidelity threshold gates the terminal state (dropped 2026-07-21).

The local eval_T / eval_stats / eval_T_and_F copies were proved bit-identical
to mc_eval and deleted (2026-07-27); call mc_eval_stats directly.
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import json, os
import numpy as np


def build_policies(ckpt):
    """{name: policy_fn(env, obs)} for agent / swap-ASAP / purify-then-swap."""
    from experiments import mc_eval as ob
    from rl_stack import policies
    return {
        "agent":       ob.make_agent_fn(ckpt),
        "swap_asap":   lambda env, obs: policies.swap_asap(env),
        "purify_swap": lambda env, obs: policies.purify_then_swap(env),
    }


def action_fractions(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps, seed=42):
    """Fraction of [NOOP, SWAP, PURIFY] among interior-node decisions over
    mc_eps greedy rollouts (source/dest excluded; the three sum to 1)."""
    from rl_stack.env_wrapper import QRNEnv
    rng = np.random.default_rng(seed)
    counts = np.zeros(3, dtype=np.int64)
    for _ in range(mc_eps):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     F0=1.0, channel_loss=0.0, max_steps=H,
                     rng=np.random.default_rng(int(rng.integers(2**32))))
        obs = env.reset()
        while not env.done and env.steps < H:
            a = int(policy_fn(env, obs))
            counts[a] += 1
            obs, _, done, _ = env.step(a)
            if done:
                break
    return (counts / max(counts.sum(), 1)).tolist()   # [f_noop, f_swap, f_purify]


def write_meta(args, extra=None):
    """Paper-provenance sidecar `<out>.meta.json`: the resolved CLI args plus
    git commit, date, host, and library versions. Written once at eval start so
    every figure JSON records exactly how it was produced."""
    import subprocess, sys, time, platform
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:   # cluster copy is not a git repo
        commit = "unknown"
    meta = dict(vars(args))
    meta.update(git_commit=commit, date=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                host=platform.node(), python=sys.version.split()[0],
                numpy=np.__version__,
                evaluator="mc_eval.mc_eval_stats(seed=42, paired)")
    if extra:
        meta.update(extra)
    path = os.path.splitext(args.out)[0] + ".meta.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(meta, open(path, "w"), indent=2, default=str)
    print(f"meta -> {path}")


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
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"saved -> {stem}.pdf")
