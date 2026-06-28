"""
Scheduling study: train an agent with (p_gen, p_swap) domain randomization on
small n_ch=2 chains (N=3,4, cutoff=5), then heatmap how close its *swap
scheduling* (PURIFY masked) gets to the exact swap-only optimum across the
(p_gen, p_swap) plane.

The DP optimum is purify-free, so the fair "did it learn optimal scheduling"
metric is the purify-masked agent's gap to T_opt_swaponly. We also report the
full-action agent's gap (purify lets it dip below the swap-only optimum).

Run from repo root with the venv active:
    PYTHONPATH=. python -m simulator.optimal_policy.pgen_pswap_study --both
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rl_stack import QRNAgent
from .compare_optimal import (
    make_agent_fns, load_optimal_pickle, _import_optimal_baseline,
)

# Domain-randomized training config: one curriculum-trained agent over the whole
# (p_gen, p_swap) regime, on homogeneous n_ch=2 chains (matching the optimal
# baseline's assumption). p_gen/p_swap as (lo, hi) -> per-episode uniform sample.
STUDY_CFG = dict(
    n_range=[3, 4],
    n_ch=[2],
    cutoff=5,
    p_gen=(0.3, 0.9),
    p_swap=(0.3, 0.9),
    F0=0.95,
    channel_loss=0.0,
    dt_seconds=0.0,
    topology="chain",
    max_steps=30,
)

GRID = [0.3, 0.5, 0.7, 0.9]   # p_gen / p_swap values the optimal pickles cover
COMPARE_N = (3, 4)
CUTOFF, HORIZON = 5, 30


def train_study(save_dir: str, episodes: int, seed: int) -> str:
    os.makedirs(save_dir, exist_ok=True)
    agent = QRNAgent(rng=np.random.default_rng(seed))
    print(f"[study] training sched_study: N={STUDY_CFG['n_range']} "
          f"n_ch={STUDY_CFG['n_ch']} cutoff={STUDY_CFG['cutoff']} "
          f"p_gen∈{STUDY_CFG['p_gen']} p_swap∈{STUDY_CFG['p_swap']} "
          f"episodes={episodes}")
    agent.train(episodes=episodes, curriculum=True, save_path=save_dir,
                plot=True, **STUDY_CFG)
    return os.path.join(save_dir, "policy.pth")


def heatmap_study(ckpt: str, save_dir: str, policy_dir: str, mc_eps: int) -> dict:
    """Evaluate the trained agent (full + purify-masked) vs the swap-only optimum
    at every (p_gen, p_swap) grid point where a pickle exists, for each N."""
    ob = _import_optimal_baseline()
    full_fn, swaponly_fn = make_agent_fns(ckpt)

    rows = []
    for N in COMPARE_N:
        for pg in GRID:
            for ps in GRID:
                payload = load_optimal_pickle(policy_dir, N, 2, CUTOFF, HORIZON, pg, ps)
                if payload is None:
                    continue
                acts = [np.asarray(a, dtype=int) for a in payload["acts"]]
                opt_fn = ob.optimal_policy_fn(payload["policy"], acts)
                T_opt, _ = ob.mc_eval(opt_fn, N, 2, pg, ps, CUTOFF, HORIZON, mc_eps)
                T_ag, _ = ob.mc_eval(full_fn, N, 2, pg, ps, CUTOFF, HORIZON, mc_eps)
                T_ag_so, _ = ob.mc_eval(swaponly_fn, N, 2, pg, ps, CUTOFF, HORIZON, mc_eps)
                rows.append({
                    "N": N, "p_gen": pg, "p_swap": ps,
                    "T_opt_swaponly": T_opt, "T_agent": T_ag,
                    "T_agent_swaponly": T_ag_so,
                    "sched_gap_pct": 100.0 * (T_ag_so - T_opt) / T_opt,
                    "gap_full_pct": 100.0 * (T_ag - T_opt) / T_opt,
                })
                print(f"  N={N} pg={pg} ps={ps}: T_opt={T_opt:.2f} "
                      f"T_ag_so={T_ag_so:.2f} sched_gap={rows[-1]['sched_gap_pct']:+.1f}% "
                      f"T_ag={T_ag:.2f} gap_full={rows[-1]['gap_full_pct']:+.1f}%",
                      flush=True)

    df = pd.DataFrame(rows)
    _plot(df, "sched_gap_pct", COMPARE_N,
          "Swap-scheduling gap to optimum (PURIFY masked)\n"
          "0 = learned optimal scheduling; positive = slower than swap-only optimum",
          os.path.join(save_dir, "heatmap_sched_gap.png"))
    _plot(df, "gap_full_pct", COMPARE_N,
          "Full-agent gap to swap-only optimum (PURIFY allowed)\n"
          "negative (blue) = agent beats swap-only optimum via purification",
          os.path.join(save_dir, "heatmap_gap_full.png"))

    out = os.path.join(save_dir, "sched_study.json")
    with open(out, "w") as f:
        json.dump({"config": {"cutoff": CUTOFF, "horizon": HORIZON,
                              "n_ch": 2, "mc_eps": mc_eps},
                   "rows": rows}, f, indent=2)
    print(f"[study] saved -> {out}")
    return {"rows": rows}


def _plot(df: pd.DataFrame, value: str, Ns, suptitle: str, path: str) -> None:
    bound = max(abs(df[value].quantile(0.02)), abs(df[value].quantile(0.98)), 1.0)
    fig, axes = plt.subplots(1, len(Ns), figsize=(7 * len(Ns), 6), squeeze=False)
    for i, N in enumerate(Ns):
        ax = axes[0][i]
        sub = df[df["N"] == N]
        pivot = sub.pivot_table(index="p_swap", columns="p_gen", values=value)
        pivot = pivot.reindex(index=sorted(GRID, reverse=True), columns=sorted(GRID))
        sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, vmin=-bound, vmax=bound,
                    annot=True, fmt=".1f", linewidths=0.5, linecolor="white",
                    cbar_kws={"label": f"{value} (%)"})
        ax.set_title(f"N = {N}", fontsize=12, fontweight="bold")
        ax.set_xlabel("$p_{gen}$")
        ax.set_ylabel("$p_{swap}$")
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="p_gen×p_swap scheduling study")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--save_dir", default="game/results/sched_study")
    ap.add_argument("--policy_dir", default="results/optimal/optimal_policies")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--mc_eps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.save_dir, exist_ok=True)
    do_train = args.train or args.both or not args.heatmap
    do_heat = args.heatmap or args.both or not args.train
    ckpt = os.path.join(args.save_dir, "policy.pth")
    if do_train:
        ckpt = train_study(args.save_dir, args.episodes, args.seed)
    if do_heat:
        heatmap_study(ckpt, args.save_dir, args.policy_dir, args.mc_eps)


if __name__ == "__main__":
    main()
